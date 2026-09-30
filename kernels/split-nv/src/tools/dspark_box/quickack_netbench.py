"""TCP_QUICKACK link check (CPU/TCP only, no GPU, no engine): a box-side server for the Mac's
og_serve/stepd_netbench.py client (ds41-stepd) that reads frames with the real split_nv.front code path -- the c1
MSG_PEEK spin, the frame head via front.recv_exact (never armed), then qa = front.quickack(plen) for the header and
payload, exactly as Front.handle_conn -- busy-waits the client's step time and replies with its reply size.

  box:  SPLIT_NV_DIR=/tmp/qa SPLIT_NV_QUICKACK=0 python3 tools/dspark_box/quickack_netbench.py 10097 300 &
        SPLIT_NV_DIR=/tmp/qa python3 tools/dspark_box/quickack_netbench.py 10098 300 &
  Mac:  /usr/bin/python3 og_serve/stepd_netbench.py run --host 10.10.10.1 --ports 10097,10098 \
            --payloads 20,92172,153600,393216 --sndbuf 1048576
2026-09-30 (DSPARK-FOLLOWUP-BOX.md s2.2): box upload p50 off -> on: 92 KB 1.134 -> 0.228 ms, 154 KB 1.277 -> 0.242,
393 KB 1.960 -> 0.229; 20 B 0.002 both. SPLIT_NV_DIR keeps the live flags file away from /dev/shm/split-nv.
"""
import os
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hooks"))
from split_nv import front as F  # noqa: E402

REPLY = struct.Struct('<dI')


def spin(conn, limit=0.06):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        try:
            conn.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            return
        except (BlockingIOError, InterruptedError):
            continue


def serve_conn(conn):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    F.keepalive(conn)
    try:
        while True:
            spin(conn)
            head = F.recv_exact(conn, F.FRAME.size)
            t_first = time.perf_counter()
            if head is None:
                return
            tag, hlen, plen = F.FRAME.unpack(head)
            qa = F.quickack(plen)
            hdr = F.recv_exact(conn, hlen, qa) if hlen else b''
            if plen:
                F.recv_exact(conn, plen, qa)
            t_full = time.perf_counter()
            if tag == b'CLOS':
                return
            step_ms, reply_bytes = struct.unpack_from('<fI', hdr)
            end = t_full + step_ms / 1000.0
            while time.perf_counter() < end:
                pass
            h = REPLY.pack(t_full - t_first, plen)
            conn.sendall(F.FRAME.pack(b'STPR', len(h), reply_bytes) + h + bytes(reply_bytes))
    finally:
        conn.close()


def main():
    port, seconds = int(sys.argv[1]), float(sys.argv[2])
    host = sys.argv[3] if len(sys.argv) > 3 else "10.10.10.1"
    print("QUICKACK", F.QUICKACK, "min", F.QUICKACK_MIN, flush=True)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    srv.settimeout(5)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            c, _ = srv.accept()
        except socket.timeout:
            continue
        threading.Thread(target=serve_conn, args=(c,), daemon=True).start()


if __name__ == "__main__":
    main()
