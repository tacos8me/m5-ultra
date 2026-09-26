"""A fake split-nv step service (STEP_API.md v1) for CPU-only failover tests.

Every STPR row carries sha256 of the session's box-side token sequence up to and
including that row (first 32 bytes of the row), so a client can check that a
rebuilt session holds exactly the tokens the original one did. Controls:
kill() drops every connection (engine crash), down()/up() stop and restart
listening (engine restart), err_opens makes OPEN answer ERR.
"""
import hashlib
import json
import socket
import struct
import threading
import time

FRAME = struct.Struct('<4sIQ')
STEP = struct.Struct('<IIH')
STPR = struct.Struct('<IIHIf')
ROW_BYTES = 40960 + 16 + 288 + 68
IDENTITY = 'split-nv:sglang-757e8f35+hooks:fp8-original:enc0-20'


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


class FakeBox:
    def __init__(self, port, step_delay=0.0):
        self.port, self.step_delay = port, step_delay
        self.sessions, self.conns, self.log = {}, [], []
        self.err_opens = False
        self.next_sid = 1
        self.srv = None
        self.lock = threading.Lock()
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

    def _serve(self, conn):
        sid = None
        try:
            while True:
                head = recv_exact(conn, FRAME.size)
                if head is None:
                    return
                tag, hlen, plen = FRAME.unpack(head)
                header = recv_exact(conn, hlen) if hlen else b''
                payload = recv_exact(conn, plen) if plen else b''
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
                    self.log.append(('open', sid, len(tokens), h.get('state')))
                    send_frame(conn, b'ACK ', dict(ok=True, session=sid, prefill_s=0.001))
                    if h.get('state', 'full') != 'none':
                        blob = state_blob(tokens)
                        send_frame(conn, b'STAT', dict(format='ds41-encoder-state-v1', prompt_tokens=len(tokens),
                                                       bytes=len(blob), prefill_s=0.001), blob)
                elif tag == b'STEP':
                    s, keep, n = STEP.unpack(header)
                    ids = list(struct.unpack('<%dI' % n, payload))
                    history = self.sessions[s]
                    if s != sid or keep > len(history):
                        send_frame(conn, b'ERR ', dict(error='bad step'))
                        return
                    history[keep:] = ids
                    self.log.append(('step', s, keep, n))
                    if self.step_delay:
                        time.sleep(self.step_delay)
                    rows = []
                    for i in range(n):
                        row = bytearray(ROW_BYTES)
                        row[:32] = digest(history[:keep + i + 1])
                        rows.append(bytes(row))
                    out = b''.join(rows)
                    conn.sendall(FRAME.pack(b'STPR', STPR.size, len(out)) + STPR.pack(s, keep + n, n, len(out), 0.001) + out)
                elif tag == b'CLOS':
                    return
        except OSError:
            return
        finally:
            if sid is not None:
                self.sessions.pop(sid, None)
            conn.close()
