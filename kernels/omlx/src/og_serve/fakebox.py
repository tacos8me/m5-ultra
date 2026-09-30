"""A fake split-nv step service (STEP_API.md v1 + STEPD-SPEC.md v1) for CPU-only failover tests.

Every STPR/STPD row carries sha256 of the session's box-side token sequence up to and
including that row (first 32 bytes of the row), so a client can check that a
rebuilt session holds exactly the tokens the original one did. Controls:
kill() drops every connection (engine crash), down()/up() stop and restart
listening (engine restart), err_opens makes OPEN answer ERR, err_steps makes every
STEP answer ERR retry:false (like "context length exceeded"), open_delay holds an
OPEN's reply that long (a long prefill; a client that goes away is logged as
('open_aborted', ...) as soon as its connection closes).

DSpark on the box (STEPD-SPEC.md, FROZEN v1 + errata E1 + v2 additions; off unless dspark is set, so older tests see
today's box): dspark=None: the box predates the spec (ACK has neither key); 'on': grant; any other string: ACK
dspark_off with that reason. v2=True (default): STEPD as b"DSTP" or b"STEP" + 18-byte header (V2.1), a plain STEP may
carry taps (V2.2), a mode 0 fallback at >= 1024 answers the [anchor, anchor] filler (V2.3); v2=False: a v1 box. kill_switch (§12), pf_active (§13), ring_reject (next RING fails its sha256, soft), err_stepd (the next STEPD
gets that hard ERR code), step_delay. draft_fn(history, keep, anchor) -> (drafts[4], maxprob[4]); tap_ok(pos, bytes)
and key_ok(stage, pos, bytes) check the ring contents of every box-drafted cycle (logged as ('draft', ...)).
Ring slots hold (position, kind, row bytes); the "keys" of tap rows are the tap bytes themselves.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import select
import socket
import struct
import threading
import time

FRAME = struct.Struct('<4sIQ')
STEP = struct.Struct('<IIH')
STPR = struct.Struct('<IIHIf')
ROW_BYTES = 40960 + 16 + 288 + 68
IDENTITY = 'split-nv:sglang-757e8f35+hooks:fp8-original:enc0-20'
CONTEXT = 1048576

_spec = importlib.util.spec_from_file_location(
    'fakebox_dspark_wire', Path(__file__).resolve().parent.parent / 'omlx/patches/deepseek_v41/dspark_wire.py')
dw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dw)


def digest(tokens):
    return hashlib.sha256(struct.pack('<%dI' % len(tokens), *tokens)).digest()


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def send_frame(conn, tag, header, payload=b''):
    h = header if isinstance(header, bytes) else json.dumps(header, separators=(',', ':')).encode()
    conn.sendall(FRAME.pack(tag, len(h), len(payload)) + h + payload)


def state_blob(tokens):
    manifest = dict(identity=IDENTITY, token_sha256=hashlib.sha256(struct.pack('<%dI' % len(tokens), *tokens)).hexdigest(),
                    bytes=0, format='ds41-encoder-state-v1')
    header = json.dumps({'__metadata__': {'manifest': json.dumps(manifest)}}).encode()
    return struct.pack('<Q', len(header)) + header


def rows_for(history, keep, n):
    out = []
    for i in range(n):
        row = bytearray(ROW_BYTES)
        row[:32] = digest(history[:keep + i + 1])
        out.append(bytes(row))
    return b''.join(out)


def default_draft(history, keep, anchor):
    h = hashlib.sha256(struct.pack('<II', keep, anchor)).digest()
    return [int.from_bytes(h[4 * i:4 * i + 4], 'little') % 1000 for i in range(4)], [0.9, 0.8, 0.7, 0.6]


class Ring:
    """One session's box ring (§2.1): slot = position % 128 -> (position, kind, bytes); ring_offset; ring_ok."""

    def __init__(self):
        self.slots, self.offset, self.ok, self.rejected = {}, 0, False, False


class FakeBox:
    def __init__(self, port, step_delay=0.0, dspark=None, v2=True):
        self.port, self.step_delay = port, step_delay
        self.sessions, self.conns, self.log = {}, [], []
        self.err_opens = False
        self.err_steps = False
        self.open_delay = 0.0
        self.next_sid = 1
        self.srv = None
        self.lock = threading.Lock()
        # STEPD (off by default: today's box)
        self.dspark = dspark
        self.v2 = v2
        self.kill_switch = False
        self.pf_active = False
        self.ring_reject = False
        self.err_stepd = None
        self.draft_fn = default_draft
        self.tap_ok = self.key_ok = None
        self.received = []  # (sid or None, tag, header bytes, payload length) of every frame, in order
        self.record_bytes = False
        self.raw = {}  # conn id -> every byte received (record_bytes)
        self.up()

    def up(self):
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', self.port))
        srv.listen(16)
        self.srv = srv
        threading.Thread(target=self._accept, args=(srv,), daemon=True).start()

    def down(self):
        self.kill()
        if self.srv is not None:
            try:
                self.srv.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.srv.close()
            self.srv = None

    def kill(self):
        with self.lock:
            conns, self.conns = self.conns, []
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()

    def _accept(self, srv):
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with self.lock:
                self.conns.append(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _recv(self, conn, n):
        data = recv_exact(conn, n)
        if data is not None and self.record_bytes:
            self.raw.setdefault(id(conn), bytearray()).extend(data)
        return data

    def _serve(self, conn):
        sid = None
        sess = None
        try:
            while True:
                head = self._recv(conn, FRAME.size)
                if head is None:
                    return
                tag, hlen, plen = FRAME.unpack(head)
                header = self._recv(conn, hlen) if hlen else b''
                payload = self._recv(conn, plen) if plen else b''
                self.received.append((sid, tag, bytes(header), plen))
                if tag == b'OPEN':
                    h = json.loads(header)
                    tokens = list(struct.unpack('<%dI' % (plen // 4), payload))
                    if self.err_opens:
                        busy = self.err_opens == 'busy'
                        send_frame(conn, b'ERR ', dict(error='fake busy' if busy else 'fake open refused', retry=busy))
                        return
                    with self.lock:
                        sid, self.next_sid = self.next_sid, self.next_sid + 1
                        self.sessions[sid] = tokens[:-1]
                    sess = dict(pending=None, granted=False, costs=None, ring=Ring())
                    self.log.append(('open', sid, len(tokens), h.get('state'), 'dspark' in h))
                    if self.open_delay:
                        deadline = time.monotonic() + self.open_delay
                        while time.monotonic() < deadline:
                            if select.select([conn], [], [], 0.05)[0] and not conn.recv(1, socket.MSG_PEEK):
                                self.log.append(('open_aborted', sid))
                                return
                    ack = dict(ok=True, session=sid, prefill_s=0.001)
                    if 'dspark' in h and self.dspark is not None:
                        costs, why = dw.parse_open(h)
                        why = why or ('disabled' if self.kill_switch else None if self.dspark == 'on' else self.dspark)
                        if why is None:
                            ack['dspark'] = dw.capability()
                            if not self.v2:
                                ack['dspark'] = {k: v for k, v in ack['dspark'].items() if k not in ('step_taps', 'filler')}
                            sess.update(granted=True, costs=costs)
                        else:
                            ack['dspark_off'] = why
                    send_frame(conn, b'ACK ', ack)
                    if h.get('state', 'full') != 'none':
                        blob = state_blob(tokens)
                        send_frame(conn, b'STAT', dict(format='ds41-encoder-state-v1', prompt_tokens=len(tokens),
                                                       bytes=len(blob), prefill_s=0.001), blob)
                elif (tag == dw.TAG_STEPD or (tag == b'STEP' and self.v2 and sess and sess['granted'])) \
                        and hlen == dw.STEPD_HDR.size:
                    if tag == b'STEP':
                        self.log.append(('stepd_alias', sid))
                    if not self._stepd(conn, sid, sess, header, payload):
                        return
                elif tag == b'STEP' and hlen == STEP.size:
                    s, keep, n = STEP.unpack(header)
                    history = self.sessions[s]
                    extra = plen - 4 * n
                    if extra and self.v2 and sess['granted'] and extra % dw.TAP_ROW_BYTES == 0 and s == sid:
                        if not self._step_taps(conn, sid, sess, keep, list(struct.unpack_from('<%dI' % n, payload)),
                                               payload[4 * n:]):
                            return
                        continue
                    if s != sid or keep > len(history) or extra:
                        send_frame(conn, b'ERR ', dict(error='bad step'))
                        return
                    ids = list(struct.unpack('<%dI' % n, payload))
                    if self.err_steps:
                        self.log.append(('step_refused', s, keep, n))
                        send_frame(conn, b'ERR ', dict(error=f'context length exceeded at {keep + n}', retry=False))
                        return
                    history[keep:] = ids
                    sess['pending'] = (keep, ids)
                    self.log.append(('step', s, keep, n))
                    if self.step_delay:
                        time.sleep(self.step_delay)
                    out = rows_for(history, keep, n)
                    conn.sendall(FRAME.pack(b'STPR', STPR.size, len(out)) + STPR.pack(s, keep + n, n, len(out), 0.001) + out)
                elif tag == b'RING':
                    if not self._ring(conn, sid, sess, json.loads(header), payload):
                        return
                elif tag == b'CLOS':
                    self.log.append(('close', sid))
                    return
                else:
                    send_frame(conn, b'ERR ', dict(error=f'unknown frame {tag!r}/{hlen}', retry=False))
                    return
        except OSError:
            return
        finally:
            if sid is not None:
                self.sessions.pop(sid, None)
            conn.close()

    # ---- STEPD-SPEC.md v1 ------------------------------------------------------------------------------------------
    def _hard(self, conn, sid, code, msg):
        self.log.append(('stepd_err', sid, code))
        send_frame(conn, b'ERR ', dict(error=f'{code}: {msg}', retry=False, code=code))
        return False

    def _ring(self, conn, sid, sess, h, payload):
        try:
            ring = dw.decode_ring(h, payload, CONTEXT)
        except dw.WireError as exc:
            return self._hard(conn, sid, exc.code, str(exc))
        if not sess['granted']:
            return self._hard(conn, sid, 'dspark_not_granted', 'RING')
        if ring['session'] != sid:
            return self._hard(conn, sid, 'bad_session', 'RING')
        r = sess['ring']
        self.log.append(('ring', sid, ring['offset'], ring['keys_rows'], ring['taps_rows']))
        if self.kill_switch:
            r.ok = False
            return True
        if self.ring_reject or ring['digest_ok'] is False:
            self.ring_reject = False
            r.ok, r.rejected = False, True
            return True
        r.slots, r.rejected = {}, False
        p, kr, tr = ring['offset'], ring['keys_rows'], ring['taps_rows']
        keys = bytes(ring['keys'])
        for stage in range(dw.STAGES):
            for j in range(kr):
                pos = p - kr - tr + j
                at = (stage * kr + j) * dw.KEY_ROW_BYTES
                r.slots.setdefault(pos % dw.RING, ['key', pos, [None] * dw.STAGES])[2][stage] = keys[at:at + dw.KEY_ROW_BYTES]
        taps = bytes(ring['taps'])
        for j in range(tr):
            pos = p - tr + j
            r.slots[pos % dw.RING] = ['tap', pos, taps[j * dw.TAP_ROW_BYTES:(j + 1) * dw.TAP_ROW_BYTES]]
        r.offset, r.ok = p, True
        return True

    def _step_taps(self, conn, sid, sess, keep, ids, taps):
        """V2.2: STEP(keep, ids) plus the ring append of the committed rows [keep - n, keep); STPR reply."""
        history = self.sessions[sid]
        n = len(taps) // dw.TAP_ROW_BYTES
        base = sess['pending'][0] if sess['pending'] else len(history)
        if not base <= keep <= len(history) or n > keep:
            return self._hard(conn, sid, 'bad_keep', f'STEP+taps keep {keep}')
        r = sess['ring']
        append = not self.kill_switch and r.ok and r.offset == keep - n
        if append:
            for j in range(n):
                pos = keep - n + j
                r.slots[pos % dw.RING] = ['tap', pos, bytes(taps[j * dw.TAP_ROW_BYTES:(j + 1) * dw.TAP_ROW_BYTES])]
            r.offset = keep
        elif r.ok:
            r.ok = False
        history[keep:] = ids
        sess['pending'] = (keep, ids)
        self.log.append(('step_taps', sid, keep, len(ids), n, append))
        out = rows_for(history, keep, len(ids))
        conn.sendall(FRAME.pack(b'STPR', STPR.size, len(out)) + STPR.pack(sid, keep + len(ids), len(ids), len(out), 0.001) + out)
        return True

    def _ring_good(self, r, keep):
        """Rows of [keep-128, keep) the ring holds with the expected contents (tap_ok / key_ok)."""
        good = 0
        for pos in range(max(0, keep - dw.RING), keep):
            slot = r.slots.get(pos % dw.RING)
            if slot is None or slot[1] != pos:
                continue
            if slot[0] == 'tap':
                good += self.tap_ok is None or self.tap_ok(pos, slot[2])
            else:
                good += all(self.key_ok is None or self.key_ok(s, pos, slot[2][s]) for s in range(dw.STAGES))
        return good

    def _stepd(self, conn, sid, sess, header, payload):
        history = self.sessions[sid]
        try:
            d = dw.decode_stepd(header, payload, CONTEXT)
        except dw.WireError as exc:
            return self._hard(conn, sid, exc.code, str(exc))
        if not sess['granted']:
            return self._hard(conn, sid, 'dspark_not_granted', 'STEPD')
        if d['session'] != sid:
            return self._hard(conn, sid, 'bad_session', 'STEPD')
        if self.err_stepd:
            code, self.err_stepd = self.err_stepd, None
            return self._hard(conn, sid, code, 'injected')
        keep, anchor, nver, a, mode = d['keep'], d['anchor'], d['nver'], d['a'], d['mode']
        if nver:
            try:
                a_box = dw.check_accept(sess['pending'], keep, anchor, nver, a, d['argmax'])
            except dw.WireError as exc:
                return self._hard(conn, sid, exc.code, str(exc))
        else:
            a_box = dw.A_BOX_NONE
            base = sess['pending'][0] if sess['pending'] else len(history)
            if not base <= keep <= len(history):
                return self._hard(conn, sid, 'bad_keep', f'kickoff keep {keep} (base {base}, length {len(history)})')
        if keep > len(history):
            return self._hard(conn, sid, 'bad_keep', f'keep {keep} > length {len(history)}')
        r = sess['ring']
        app = d['ntaps']
        off = self.kill_switch  # read once per job, as the box does (§12: carried in the job command)
        if app:  # §6.3
            if off:
                r.ok = False
            elif r.ok and r.offset == keep - app:
                taps = bytes(d['taps'])
                for j in range(app):
                    pos = keep - app + j
                    r.slots[pos % dw.RING] = ['tap', pos, taps[j * dw.TAP_ROW_BYTES:(j + 1) * dw.TAP_ROW_BYTES]]
                r.offset = keep
            else:
                r.ok = False
        elif nver and d['flags'] & dw.F_NO_TAPS:
            r.ok = False
        ring_ok = r.ok and r.offset == keep
        dmax = min(d['dmax'], max(0, CONTEXT - keep - 1))
        reasons = [(off, 1), (r.rejected, 5), (not ring_ok, 2),
                   (mode == dw.MODE_BOX and keep < dw.MIN_CTX, 3), (mode == dw.MODE_BOX and dmax < 1, 4)]
        reason = next((code for hit, code in reasons if hit), 0)
        drafts, probs, used, drafter_ms = [], [], mode, 0.0
        if mode == dw.MODE_BOX:
            if reason == 0:
                drafts, probs = self.draft_fn(list(history[:keep]), keep, anchor)
                depth = dw.box_depth(probs, sess['costs'], keep, bool(d['flags'] & dw.F_FUSED), dmax)
                ids = [anchor] + list(drafts[:depth])
                drafter_ms = 1.5
                self.log.append(('draft', sid, keep, self._ring_good(r, keep), min(dw.RING, keep), depth))
            elif self.v2 and reason in (1, 2, 5) and keep >= dw.MIN_CTX and dmax >= 1:
                ids, drafts, probs = [anchor, anchor], [anchor] * 4, [0.0] * 4  # V2.3 filler
                self.log.append(('filler', sid, keep, reason))
            else:
                used, ids = dw.MODE_BONUS, [anchor]
                self.log.append(('bonus', sid, keep, reason))
        elif mode == dw.MODE_EXPLICIT:
            ids = [anchor] + d['explicit']
            if reason not in (1, 2, 5):
                reason = 0
        else:
            ids = [anchor]
        history[keep:] = ids
        sess['pending'] = (keep, ids)
        self.log.append(('stepd', sid, keep, nver, a, mode, used, len(ids), app))
        if self.step_delay:
            time.sleep(self.step_delay)
        flags = ((dw.R_DRAFTED if drafter_ms else 0) | (dw.R_PF_ACTIVE if self.pf_active else 0)
                 | (dw.R_RING_OK if ring_ok and not off else 0) | (dw.R_DISABLED if off else 0))
        out = rows_for(history, keep, len(ids))
        hdr = dw.encode_stpd_header(sid, keep, len(ids), 0.002, a_box, used, flags, reason,
                                    drafts if used == dw.MODE_BOX else (), probs if used == dw.MODE_BOX else (),
                                    drafter_ms)
        conn.sendall(FRAME.pack(dw.TAG_STPD, len(hdr), len(out)) + hdr + out)
        return True
