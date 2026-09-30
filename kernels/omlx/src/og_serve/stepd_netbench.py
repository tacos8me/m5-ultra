"""STEPD upload microbench: what ~100 KB of taps per cycle costs on the Mac -> box link (CPU and network only).

No MLX, no GPU, no engine: a box-side server that reads frames the way split_nv/front.py does (MSG_PEEK spin, then
recv_exact), busy-waits a fixed "step" and answers with an STPR-sized reply; a Mac-side client that runs c1-shaped
cycles (Mac work spin -> send frame -> spin for the reply) on a fresh connection per "request".

  box:  python3 og_serve/stepd_netbench.py serve 10095            # today's front at c1 (MSG_PEEK spin)
        python3 og_serve/stepd_netbench.py serve 10096 --quickack  # TCP_QUICKACK before every recv
        python3 og_serve/stepd_netbench.py serve 10097 --no-spin   # today's front at c2+ (blocking recv)
  Mac:  /usr/bin/python3 og_serve/stepd_netbench.py run --ports 10095,10096 --payloads 20,92172,153600 [--sndbuf 1048576]

Per cell (port:payload): median / p90 of rtt (send start -> reply received), send (sendall), upload (box: first frame
byte -> last payload byte) and payload_rx (Mac: reply header -> last byte). rtt - step_ms is the link + host cost.
2026-09-30 (DSPARK-FOLLOWUP-MAC.md): upload 1.1-1.5 ms (p90 1.9-2.5) for 92-154 KB on today's socket, 0.08-0.22 ms
(p90 0.25) with TCP_QUICKACK; 20 B: 0.002 ms.
"""
import argparse
import select
import socket
import statistics
import struct
import threading
import time

FRAME = struct.Struct('<4sIQ')
REPLY = struct.Struct('<dI')  # upload seconds (box: first byte -> last byte), payload bytes seen


def recv_exact(conn, n, quickack=False):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        if quickack:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)  # not sticky: before every read
        k = conn.recv_into(view[got:], min(n - got, 4 << 20))
        if k == 0:
            return None
        got += k
    return buf


def spin_readable(conn, limit=0.06):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        try:
            conn.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            return
        except (BlockingIOError, InterruptedError):
            continue


def serve_conn(conn, quickack, spin):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            if spin:  # front.py spins like this only while a single session decodes (c1)
                spin_readable(conn)
            head = recv_exact(conn, FRAME.size, quickack)
            t_first = time.perf_counter()
            if head is None:
                return
            tag, hlen, plen = FRAME.unpack(head)
            hdr = recv_exact(conn, hlen, quickack) if hlen else b''
            if plen:
                recv_exact(conn, plen, quickack)
            t_full = time.perf_counter()
            if tag == b'CLOS':
                return
            step_ms, reply_bytes = struct.unpack_from('<fI', hdr)
            end = t_full + step_ms / 1000.0
            while time.perf_counter() < end:
                pass
            h = REPLY.pack(t_full - t_first, plen)
            conn.sendall(FRAME.pack(b'STPR', len(h), reply_bytes) + h + bytes(reply_bytes))
    finally:
        conn.close()


def serve(args):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if args.rcvbuf:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, args.rcvbuf)
    srv.bind((args.host, args.port))
    srv.listen(8)
    srv.settimeout(5)
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        threading.Thread(target=serve_conn, args=(conn, args.quickack, not args.no_spin), daemon=True).start()


def receive(sock, size):
    buf = bytearray(size)
    view = memoryview(buf)
    off = 0
    while off < size:
        n = sock.recv_into(view[off:], min(size - off, 4 << 20))
        if not n:
            raise ConnectionError('closed')
        off += n
    return buf


def one_request(args, port, payload):
    s = socket.create_connection((args.host, port), timeout=5)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)  # pipe_wire.RCVBUF
    if args.sndbuf:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, args.sndbuf)
    body = bytes(payload)
    hdr = struct.pack('<fI', args.step_ms, args.reply) + bytes(10)  # 18 B, the STEPD header size
    rows = []
    for _ in range(args.cycles):
        end = time.perf_counter() + args.mac_ms / 1000.0
        while time.perf_counter() < end:
            pass
        t0 = time.perf_counter()
        s.sendall(FRAME.pack(b'DSTP', len(hdr), len(body)) + hdr + body)
        t_sent = time.perf_counter()
        deadline = t_sent + 0.06  # pipe_wire's recv spin
        while time.perf_counter() < deadline and not select.select([s], [], [], 0)[0]:
            pass
        _, hl, pl = FRAME.unpack(receive(s, FRAME.size))
        upload, _ = REPLY.unpack(receive(s, hl))
        t_hdr = time.perf_counter()
        receive(s, pl)
        t_done = time.perf_counter()
        rows.append(dict(rtt=(t_done - t0) * 1e3, send=(t_sent - t0) * 1e3, upload=upload * 1e3,
                         payload_rx=(t_done - t_hdr) * 1e3))
    s.sendall(FRAME.pack(b'CLOS', 0, 0))
    s.close()
    return rows[args.warmup:]


def run(args):
    ports = [int(p) for p in args.ports.split(',')]
    payloads = [int(p) for p in args.payloads.split(',')]
    cells = {}
    for _ in range(args.reps):
        for port in ports:
            for payload in payloads:
                cells.setdefault((port, payload), []).extend(one_request(args, port, payload))
    for (port, payload), rows in cells.items():
        def pick(q):
            return {k: round(sorted(r[k] for r in rows)[min(len(rows) - 1, int(q * len(rows)))], 3) for k in rows[0]}
        med = {k: round(statistics.median(r[k] for r in rows), 3) for k in rows[0]}
        print(f'{port}:{payload} n {len(rows)} median {med} p90 {pick(0.9)}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sv = sub.add_parser('serve')
    sv.add_argument('port', type=int)
    sv.add_argument('--host', default='10.10.10.1')
    sv.add_argument('--rcvbuf', type=int, default=0)
    sv.add_argument('--quickack', action='store_true')
    sv.add_argument('--no-spin', action='store_true', help='blocking recv between frames (front.py at c2+)')
    sv.add_argument('--seconds', type=float, default=600)
    rn = sub.add_parser('run')
    rn.add_argument('--host', default='10.10.10.1')
    rn.add_argument('--ports', default='10095')
    rn.add_argument('--payloads', default='20,92172,153600')
    rn.add_argument('--cycles', type=int, default=120)
    rn.add_argument('--warmup', type=int, default=5)
    rn.add_argument('--reps', type=int, default=2)
    rn.add_argument('--mac-ms', type=float, default=17.0)
    rn.add_argument('--step-ms', type=float, default=7.0)
    rn.add_argument('--reply', type=int, default=5 * 41332)
    rn.add_argument('--sndbuf', type=int, default=0)
    args = ap.parse_args()
    serve(args) if args.cmd == 'serve' else run(args)


if __name__ == '__main__':
    main()
