"""Mac-side STEP exchange overhead (no GPU): roundtrip - box_s, payload time, per socket option set.

Opens one lean box session on the 8K prompt, then STEPs 1 and 5 rows at a fixed keep (rewinds each time).
Variants toggle per-socket options only (no sysctls). Run only while production is idle.
"""
import json, os, socket, statistics, sys, time
from pathlib import Path
sys.path.insert(0, os.environ.get('DS41_TREE', str(Path.home()/'src/wt/ds41-host')))
from omlx.patches.deepseek_v41 import pipe_wire as pw

HOME = Path.home()
tokens = json.loads((HOME/'llm/ds41/split-wire/ids-8192.json').read_text())
N = int(os.environ.get('WIRE_STEPS', '60'))
variants = os.environ.get('WIRE_VARIANTS', 'base,rcvbuf,base,rcvbuf').split(',')
SLEEP = float(os.environ.get('WIRE_SLEEP_MS', '0')) / 1000
ADVANCE = os.environ.get('WIRE_ADVANCE', '0') == '1'
GAP = float(os.environ.get('WIRE_GAP_MS', '0')) / 1000  # idle before each send (the Mac forward + draft)
TCP_SENDMOREACKS = 0x10A


def apply(sock, v):
    if v in ('rcvbuf', 'both'):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    if v in ('moreacks', 'both'):
        sock.setsockopt(socket.IPPROTO_TCP, TCP_SENDMOREACKS, 1)
    if v == 'base':
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)


enc = pw.EncoderSession()
t = time.perf_counter()
tensors, manifest = enc.open(tokens, 'host-wire', cache=True, state='lean', stream=True)
print(json.dumps(dict(event='open', s=round(time.perf_counter() - t, 2))), flush=True)
keep = enc.length
out = []
try:
    for v in variants:
        apply(enc.sock, v)
        for rows in (1, 5):
            recs = []
            for i in range(N + 3):
                ids = [tokens[-1]] + [tokens[(i * 7 + j) % 8000] for j in range(rows - 1)]
                if ADVANCE and i:
                    keep = min(enc.length, keep + 1 + (i % rows))  # commit 1..rows rows, like accept+rollback
                if GAP:
                    end = time.perf_counter() + GAP
                    while time.perf_counter() < end:
                        pass
                enc.send_step(ids, keep)
                if SLEEP:
                    time.sleep(SLEEP)
                raw, tm = enc.recv_step()
                if i >= 3:
                    recs.append(tm)
            m = lambda k: round(1000 * statistics.median(r[k] for r in recs), 3)
            rec = dict(variant=v, advance=ADVANCE, gap_ms=GAP * 1000, rows=rows, box_ms=m('box_s'), roundtrip_ms=m('roundtrip_s'),
                       overhead_ms=round(1000 * statistics.median(r['roundtrip_s'] - r['box_s'] for r in recs), 3),
                       payload_ms=m('payload_s'), rcvbuf=enc.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
            print(json.dumps(rec), flush=True)
            out.append(rec)
finally:
    enc.close()
