"""CPU-only checks of the step front end's connection hygiene (no GPU: CUDA_VISIBLE_DEVICES is emptied first).

  1. accepted step connections carry TCP keepalive (a Mac that vanished without a FIN is noticed and its sessions,
     with their KV, are closed instead of staying allocated forever);
  2. an oversized frame is refused with ERR before anything is allocated for it;
  3. a vanished peer's session is closed once keepalive gives up (Linux: the blocked recv fails);
  4. /health carries stalest_step_s and gpu_mem_mib.
usage: python3 tools/test_front_robust.py
"""
import os
import socket
import struct
import sys
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hooks"))
import split_nv.front as F  # noqa: E402


class Engine:
    def __init__(self):
        self.sessions = {}
        self.vision = False


class Cache:
    def summary(self):
        return {}


def front():
    fr = object.__new__(F.Front)
    fr.engine = Engine()
    fr.open_conns, fr.conn_lock = 0, threading.Lock()
    fr.last_step, fr.current, fr.draining = {}, None, False
    fr.closed = []
    fr.submit = lambda cmd, priority=0: fr.closed.append(cmd)
    fr.spin_readable = lambda conn, sid: None
    fr.step_port, fr.version, fr.t_start = 1, "test", time.time()
    fr.cache = Cache()
    fr.jobs = type("Q", (), {"qsize": lambda self: 0})()
    # fair (preempt/bypass/trim) attributes that Front.__init__ sets; defaults = all fair features off
    fr.prefill_lock = threading.Lock()
    fr.gate = F.PrefillGate(fr.prefill_lock, 0) if hasattr(F, "PrefillGate") else None
    fr.preempt_on, fr.trim_min = False, 0
    fr.pstats = {"chunks": 0, "inline_steps": 0, "after_chunk_steps": 0}
    return fr


def pair():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    cli = socket.create_connection(srv.getsockname())
    conn, peer = srv.accept()
    srv.close()
    return cli, conn, peer


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f": {detail}"), flush=True)
    return bool(cond)


def main():
    results = []
    fr = front()
    cli, conn, peer = pair()
    t = threading.Thread(target=fr.handle_conn, args=(conn, peer), daemon=True)
    t.start()
    time.sleep(0.2)
    idle = conn.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE)
    results.append(check("step connection has TCP keepalive", conn.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
                         and idle == F.KEEPALIVE_S, idle))
    cli.sendall(F.FRAME.pack(b"OPEN", 16, 8 << 30))  # claims an 8 GiB payload
    cli.settimeout(3)
    head = cli.recv(F.FRAME.size)
    tag = F.FRAME.unpack(head)[0] if len(head) == F.FRAME.size else None
    t.join(3)
    results.append(check("oversized frame -> ERR, connection closed, nothing allocated", tag == b"ERR "
                         and not t.is_alive() and fr.open_conns == 0, (tag, t.is_alive(), fr.open_conns)))
    cli.close()

    # A peer that disappears without a FIN: emulate by keepalive giving up fast on a socket whose peer stops
    # answering is not possible on loopback, so check the mechanism instead: the blocked recv of handle_conn
    # returns once the socket errors (shutdown from outside == what the kernel does when keepalive fails).
    fr = front()
    fr.engine.sessions[7] = object()
    cli, conn, peer = pair()
    t = threading.Thread(target=fr.handle_conn, args=(conn, peer), daemon=True)
    t.start()
    time.sleep(0.2)
    conn.shutdown(socket.SHUT_RD)
    t.join(3)
    results.append(check("a dead connection ends its handler (session close path runs)", not t.is_alive()
                         and fr.open_conns == 0, (t.is_alive(), fr.open_conns)))
    cli.close()

    fr = front()
    fr.last_step = {1: time.monotonic() - 42.0, 2: time.monotonic()}
    ok, body = fr.health()
    results.append(check("health: stalest_step_s and gpu_mem_mib present", 41 < body["stalest_step_s"] < 44
                         and "gpu_mem_mib" in body, body))
    ok = all(results)
    print("ALL PASS" if ok else "SOME FAILED", f"({sum(results)}/{len(results)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
