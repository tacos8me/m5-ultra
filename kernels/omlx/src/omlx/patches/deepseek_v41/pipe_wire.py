"""CPU-only client for split-nv STEP_API.md v1; one connection per request."""
from array import array
import hashlib
import json
import logging
import math
import os
import select
import socket
import struct
import time
import zlib

FRAME = struct.Struct('<4sIQ')
STEP = struct.Struct('<IIH')
STPR = struct.Struct('<IIHIf')
ROW_BYTES = 40960 + 16 + 288 + 68
DTYPES = {'U8':1,'I8':1,'U16':2,'I16':2,'BF16':2,'F16':2,'U32':4,'I32':4,'F32':4,'I64':8,'U64':8}
# Steps run between another session's 8K prefill chunks (box c889811+), so a step waits at
# most ~1.5 s behind a peer's prefill; a step with no reply in this time is a hung box.
STEP_TIMEOUT = float(os.environ.get('DS41_OG_STEP_TIMEOUT_S', '120'))
# Spin (instead of sleeping in recv) while the box step is in flight: a blocked
# thread wakes on a cold core and the following MLX graph build runs ~3x slower
# (measured 7.5 -> 2.1 ms per 20-layer build, drafter 8.9 -> 7.7 ms).
SPIN_S = float(os.environ.get('DS41_OG_SPIN_MS', '60'))/1000
IDENTITY = 'split-nv:sglang-757e8f35+hooks:fp8-original:enc0-20'
# Mid-stream recovery: a lost box session (link drop, engine restart) is
# re-opened on the committed tokens with OPEN state "none" and the failed STEP
# is resent. The box's step and prefill arithmetic agree bit for bit and the
# Mac's layer 20-39 cache is untouched, so the continuation is identical.
RESUME_WAIT_S = float(os.environ.get('DS41_OG_RESUME_WAIT_S', '45'))
# A box host that answers but refuses the port is restarting its engine (systemd restart after a
# crash or a deploy: 22-52 s drain + ~2 min load, og-box); ride that out on og, not q3.
RESTART_WAIT_S = float(os.environ.get('DS41_OG_RESTART_WAIT_S', '240'))
RESUME_TRIES = int(os.environ.get('DS41_OG_RESUME_TRIES', '3'))
# A box that answers OPEN with a retryable ERR (draining, KV/session budget full) is waited for
# this long; it does not count as a failed try.
BUSY_WAIT_S = float(os.environ.get('DS41_OG_BUSY_WAIT_S', '300'))
# Rebuild through the box prefix cache (box s4+: byte-identical to a fresh prefill; older boxes
# ignore the flag), so a rebuild after a link drop restores the prompt instead of re-prefilling it.
REOPEN_CACHE = os.environ.get('DS41_OG_REOPEN_CACHE', '1') == '1'
logger = logging.getLogger(__name__)


RECOVERIES = []  # every rebuilt session (process-wide), newest last


class BoxLost(ConnectionError):
    """The box stayed unavailable past DS41_OG_RESUME_WAIT_S; the session cannot continue."""


class BoxBusy(RuntimeError):
    """The box refused an OPEN with a retryable ERR (draining, or no KV/session room yet)."""


def wait_limit(exc):
    """How long a rebuild/open keeps retrying after this failure."""
    if isinstance(exc, BoxBusy):
        return BUSY_WAIT_S
    if isinstance(exc, ConnectionRefusedError):
        return RESTART_WAIT_S
    return RESUME_WAIT_S


def refused(tag, header, what):
    if tag == b'ERR ' and isinstance(header, dict) and header.get('retry'):
        return BoxBusy(f'{what} busy: {header.get("error")}')
    return RuntimeError(f'{what} refused: {str(header)[:200]}')


def receive(sock, size):
    buf = bytearray(size)
    view = memoryview(buf)
    offset = 0
    while offset < size:
        count = sock.recv_into(view[offset:], min(size-offset, 4<<20))
        if not count:
            raise ConnectionError('Encoder connection closed')
        offset += count
    return buf


def send(sock, tag, header, payload=b''):
    h = header if isinstance(header, bytes) else json.dumps(header, separators=(',', ':')).encode()
    sock.sendall(FRAME.pack(tag, len(h), len(payload)) + h + payload)


def frame(sock):
    tag, hlen, plen = FRAME.unpack(receive(sock, FRAME.size))
    if hlen > 4<<20 or plen > 2<<30:
        raise ValueError('Encoder frame exceeds limit')
    raw = receive(sock, hlen)
    head = bytes(raw) if tag == b'STPR' else json.loads(raw) if hlen else {}
    return tag, head, plen


class EncoderSession:
    def __init__(self, host='10.10.10.1', port=10052, *, identity=IDENTITY, timeout=120):
        self.host, self.port = host, port
        self._connect(timeout)
        self.identity, self.session, self.length = identity, None, None
        self._pending = None
        self.open_info = {}
        # Box-side token sequence (tokens[:-1] of OPEN, then every STEP's rows at `keep`):
        # the prefix a lost session is rebuilt from.
        self.history = array('I')
        self.images = []  # og_images.Image of the OPEN; a rebuilt session gets them again
        self.recoveries = []

    def _connect(self, timeout=120):
        self.sock = socket.create_connection((self.host, self.port), timeout=5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)

    def close(self):
        if self.sock is not None:
            try:
                if self.session is not None:
                    send(self.sock, b'CLOS', {'session':self.session})
            except OSError:
                pass
            self.sock.close()
            self.sock = None

    def open(self, tokens, request_id='', *, cache=False, state='full', delta_from=0, stream=False, images=None):
        """OPEN a box session; returns the (lazy) state tensors and manifest.

        cache: let the box resume from its longest snapshot that prefixes
        tokens[:-1] (ACK.resumed_tokens); the state bytes equal a fresh prefill.
        state: 'full' | 'lean' (layer 20 + tail only) | 'none'.
        delta_from=P: the caller holds layer-20 rows [0, P) of this exact
        token prefix; the box may send rows [P, N-1) only (manifest delta_from).
        stream: ACK first, then TENS parts as prefill chunks finish (same bytes).
        images: [og_images.Image]; the box runs the vision tower on their patches. With images,
        prefix_sha256 covers the uint64 prefix keys (image content), not the token ids.
        """
        if self.session is not None or len(tokens) < 2 or len(tokens) > 1048576:
            raise ValueError('Invalid session open')
        if not 0 <= delta_from < len(tokens):
            raise ValueError('Invalid delta_from')
        payload = struct.pack('<%dI' % len(tokens), *tokens)
        digest = hashlib.sha256(payload).hexdigest()
        header = dict(proto=1, identity=self.identity, prompt_tokens=len(tokens),
                      token_sha256=digest, state=state, request_id=request_id)
        if cache:
            header['cache'] = 1
        if stream:
            header['stream'] = 1
        images = list(images or ())
        tail = b''
        if images:
            from . import og_images
            header['images'], tail = og_images.wire(images)
        if delta_from:
            header['delta_from'] = int(delta_from)
            if images:
                from . import og_images
                header['prefix_sha256'] = og_images.keys_digest(og_images.prompt_keys(tokens[:delta_from], images), delta_from)
            else:
                header['prefix_sha256'] = hashlib.sha256(payload[:4*delta_from]).hexdigest()
        self.sock.settimeout(600)
        start = time.perf_counter()
        send(self.sock, b'OPEN', header, payload + tail)
        self.images = images
        tag, ack, n = frame(self.sock)
        if tag != b'ACK ' or not ack.get('ok') or n:
            raise refused(tag, ack, 'Encoder open')
        self.session = int(ack['session'])
        self.open_info = dict(resumed_tokens=int(ack.get('resumed_tokens') or 0),
                              ack_s=time.perf_counter()-start, delta_requested=int(delta_from))
        tensors, manifest, total = {}, None, 0
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            tag, h, n = frame(self.sock)
            if tag == b'STAT':
                if tensors or manifest is not None or h.get('bytes',n) != n:
                    raise ValueError('Unexpected STAT frame')
                received=time.perf_counter()
                blob=receive(self.sock,n)
                self.open_info.update(box_prefill_s=h.get('prefill_s'),state_bytes=n,
                                      transfer_s=time.perf_counter()-received)
                tensors,manifest=parse_state_blob(blob)
                if manifest.get('identity') != self.identity or manifest.get('token_sha256') != digest:
                    raise ValueError('Encoder identity/prompt mismatch')
                self.length=len(tokens)-1
                self.history=array('I',tokens[:-1])
                self.sock.settimeout(STEP_TIMEOUT)
                return tensors,manifest
            elif tag == b'TENS':
                name, dtype, shape = h['name'], h['dtype'], h['shape']
                if any(not isinstance(x,int) or x < 0 for x in shape):
                    raise ValueError('Invalid tensor shape')
                size = math.prod(shape)*DTYPES[dtype]
                if name not in tensors:
                    if total+size > 2<<30:
                        raise ValueError('Encoder state exceeds 2 GiB')
                    total += size
                    tensors[name] = [dtype, shape, bytearray(size), 0]
                entry = tensors[name]
                offset = h.get('offset',0)
                if entry[:2] != [dtype,shape] or offset != entry[3] or offset+n > size or h.get('nbytes',n)!=n:
                    raise ValueError('Invalid tensor part order/geometry')
                data = receive(self.sock,n)
                if 'crc32' in h and zlib.crc32(data) != h['crc32']:
                    raise ValueError('Tensor CRC mismatch')
                entry[2][offset:offset+n] = data
                entry[3] += n
            elif tag == b'MANI' and not n and manifest is None:
                manifest = h
            elif tag == b'PROG' and not n:
                continue
            elif tag == b'END ' and not n:
                if manifest is None or any(len(x[2]) != x[3] for x in tensors.values()):
                    raise ValueError('Incomplete encoder state')
                if h.get('bytes',total) != total or manifest.get('bytes') != total:
                    raise ValueError('Encoder byte count mismatch')
                self.open_info.update(box_prefill_s=h.get('prefill_s'), state_bytes=total)
                if manifest.get('identity') != self.identity or manifest.get('token_sha256') != digest:
                    raise ValueError('Encoder identity/prompt mismatch')
                self.length = len(tokens)-1
                self.history = array('I', tokens[:-1])
                self.sock.settimeout(STEP_TIMEOUT)
                return tensors, manifest
            else:
                raise RuntimeError('Unexpected encoder frame: '+str((tag,h)))
        raise TimeoutError('Encoder prefill deadline')

    def send_step(self, tokens, keep):
        """Queue one STEP; the box computes while the caller does other work."""
        if self.session is None or not 1 <= len(tokens) <= 5 or not 0 <= keep <= self.length:
            raise ValueError('Invalid step/rollback')
        if getattr(self, '_pending', None) is not None:
            raise ValueError('Step already in flight')
        payload = struct.pack('<%dI' % len(tokens), *tokens)
        self._pending = (len(tokens), keep, time.perf_counter(), tuple(tokens))
        self.history[keep:] = array('I', tokens)
        send(self.sock,b'STEP',STEP.pack(self.session,keep,len(tokens)),payload)

    def ensure_step(self, tokens, keep):
        """Send this STEP unless the identical one is already in flight (pre-sent)."""
        pending = self._pending
        if pending is not None and pending[1] == keep and pending[3] == tuple(tokens):
            return True
        if pending is not None:
            self.recv_step()  # a stale pre-sent step: consume its reply, then replace it
        self.send_step(tokens, keep)
        return False

    def recv_step(self):
        rows_sent, keep, start, _ = self._pending
        waited = time.perf_counter()
        deadline = waited + SPIN_S
        while time.perf_counter() < deadline and not select.select([self.sock], [], [], 0)[0]:
            pass
        tag, h, n = frame(self.sock)
        header_at = time.perf_counter()
        if tag != b'STPR' or len(h) != STPR.size:
            raise RuntimeError('Encoder step failed: '+str((tag,h)))
        session, after, rows, size, box_s = STPR.unpack(h)
        if (session,after,rows,size,n) != (self.session,keep+rows_sent,rows_sent,rows_sent*ROW_BYTES,rows_sent*ROW_BYTES):
            raise ValueError('Encoder step/session boundary mismatch')
        if not math.isfinite(box_s) or box_s < 0:
            raise ValueError('Invalid encoder timing')
        raw = receive(self.sock,n)
        self._pending = None
        self.length = after
        done = time.perf_counter()
        elapsed = done-start
        return raw, dict(rows=rows,start=keep,box_s=box_s,roundtrip_s=elapsed,wait_s=done-waited,
                         transport_host_s=elapsed-box_s,payload_s=done-header_at,bytes=n)

    def step(self, tokens, keep):
        self.send_step(tokens, keep)
        return self.recv_step()

    # ---- mid-stream recovery -------------------------------------------------------------------
    def _drop(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock, self.session, self._pending = None, None, None

    def reopen(self, keep, next_token):
        """Open a new session holding exactly history[:keep] (box prefill, OPEN state "none")."""
        if not 1 <= keep <= len(self.history):
            raise ValueError(f'cannot rebuild a box session at {keep} of {len(self.history)} tokens')
        prefix = self.history[:keep]
        payload = prefix.tobytes() + struct.pack('<I', next_token)
        self.sock.settimeout(600)
        header = dict(proto=1, identity=self.identity, prompt_tokens=keep+1,
                      token_sha256=hashlib.sha256(payload).hexdigest(), state='none', request_id='resume')
        if REOPEN_CACHE:
            header['cache'] = 1
        images = [im for im in getattr(self, 'images', ()) if im.start + im.length <= keep]
        tail = b''
        if images:
            from . import og_images
            header['images'], tail = og_images.wire(images)
        send(self.sock, b'OPEN', header, payload + tail)
        tag, ack, n = frame(self.sock)
        if tag != b'ACK ' or not ack.get('ok') or n:
            raise refused(tag, ack, 'Encoder reopen')
        self.session, self.length, self.history = int(ack['session']), keep, prefix
        self.sock.settimeout(STEP_TIMEOUT)
        return ack

    def recover(self, tokens, keep, error, since=None):
        """Rebuild a lost session at `keep` and resend STEP(tokens, keep).

        Waits for the box up to RESUME_WAIT_S from `since` (connection refused/unreachable); a box
        that accepts the connection but fails the reopen or step RESUME_TRIES times is lost too.
        """
        since = since or time.monotonic()
        failures, last = 0, error
        logger.warning('ds41-og box session %s lost at %d tokens (%r); rebuilding', self.session, keep, error)
        while True:
            self._drop()
            t0 = time.perf_counter()
            try:
                self._connect()
            except OSError as exc:
                last = exc
            else:
                try:
                    ack = self.reopen(keep, tokens[0])
                    self.send_step(tokens, keep)
                    record = dict(keep=keep, wait_s=round(time.monotonic()-since, 3), prefill_s=ack.get('prefill_s'),
                                  resumed_tokens=ack.get('resumed_tokens'),
                                  reopen_s=round(time.perf_counter()-t0, 3), error=repr(error)[:160], t=time.time())
                    self.recoveries.append(record)
                    RECOVERIES.append(record)
                    del RECOVERIES[:-100]
                    logger.warning('ds41-og box session rebuilt at %d tokens: session %d after %.1fs (box prefill %.2fs)',
                                   keep, self.session, time.monotonic()-since, ack.get('prefill_s') or 0)
                    return
                except (OSError, RuntimeError, ValueError, struct.error) as exc:
                    last, failures = exc, failures + (not isinstance(exc, BoxBusy))
            if failures >= RESUME_TRIES or time.monotonic() - since >= wait_limit(last):
                self._drop()
                raise BoxLost(f'box unavailable for {time.monotonic()-since:.0f}s at {keep} tokens: {last!r}'[:300])
            time.sleep(1.0)

    def ensure_step_safe(self, tokens, keep):
        """ensure_step() that rebuilds a lost session (returns False: nothing was pre-sent)."""
        try:
            return self.ensure_step(tokens, keep)
        except (OSError, RuntimeError, ValueError, struct.error, TypeError) as exc:
            if isinstance(exc, BoxLost):
                raise
            self.recover(tokens, keep, exc)
            return False

    def recv_step_safe(self, tokens, keep):
        """recv_step() for STEP(tokens, keep) that rebuilds a lost session and resends the step."""
        since = None
        while True:
            try:
                return self.recv_step()
            except (OSError, RuntimeError, ValueError, struct.error, TypeError) as exc:
                if isinstance(exc, BoxLost):
                    raise
                since = since or time.monotonic()
                self.recover(tokens, keep, exc, since)


def parse_state_blob(blob):
    if len(blob)<8:
        raise ValueError('Truncated safetensors')
    size=struct.unpack('<Q',blob[:8])[0]
    if size>4<<20 or size+8>len(blob):
        raise ValueError('Invalid safetensors header')
    header=json.loads(blob[8:8+size])
    manifest=json.loads(header.pop('__metadata__')['manifest'])
    tensors={};end=0;base=size+8
    for name,entry in sorted(header.items(),key=lambda x:x[1]['data_offsets'][0]):
        dtype,shape=entry['dtype'],entry['shape']
        if any(type(x) is not int or x<0 for x in shape):
            raise ValueError('Invalid tensor shape')
        a,b=entry['data_offsets'];expected=math.prod(shape)*DTYPES[dtype]
        if a!=end or b-a!=expected or base+b>len(blob):
            raise ValueError('Invalid safetensors range')
        tensors[name]=[dtype,shape,memoryview(blob)[base+a:base+b],expected]
        end=b
    if base+end!=len(blob) or manifest['bytes']!=end:
        raise ValueError('Safetensors byte count mismatch')
    return tensors,manifest


def mlx_array(dtype, shape, raw):
    import numpy as np
    import mlx.core as mx
    dt = {'BF16':'<u2','F16':'<f2','F32':'<f4','U8':'u1','I8':'i1',
          'U16':'<u2','I16':'<i2','U32':'<u4','I32':'<i4','U64':'<u8','I64':'<i8'}[dtype]
    value = mx.array(np.frombuffer(raw,dt).reshape(shape))
    return value.view(mx.bfloat16) if dtype == 'BF16' else value


def mlx_state(tensors, names=None):
    """MLX arrays for every tensor, or only for `names` (the og import skips layers 0-19)."""
    items = tensors.items() if names is None else ((name, tensors[name]) for name in names)
    return {name:mlx_array(dtype,shape,raw) for name,(dtype,shape,raw,_) in items}


def mlx_step(raw, rows):
    view, at, out = memoryview(raw), 0, {}
    for name,dtype,shape in [('h','BF16',(1,rows,4,5120)),('pre','F32',(1,rows,4)),
                            ('kv','U8',(1,rows,288)),('index','U8',(1,rows,68))]:
        size = math.prod(shape)*DTYPES[dtype]
        out[name] = mlx_array(dtype,shape,view[at:at+size])
        at += size
    if at != len(raw):
        raise ValueError('Unexpected step payload length')
    return out
