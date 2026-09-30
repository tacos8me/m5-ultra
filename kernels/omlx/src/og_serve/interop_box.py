"""Interop: this tree's Mac client (pipe_wire + stepd, driven by og_serve/test_stepd.MacSim) against the box's real
STEPD implementation: split_nv.front.Front (OPEN grant, RING, DSTP/STEP dispatch, filler, STEP+taps) and
split_nv.engine.Engine.cmd_stepd / cmd_ring / cmd_step on two ranks, with the box test's fake GPU parts (fake step
rows, a fake drafter that decodes (token, position) from the tap/key rows and drafts with a toy LM).
Source: the box tree's tools/test_dspark_stepd.py (make_front / serve / lm / enc_row / expected_payload).

CPU only, loopback only, no engine, no /dev/shm/split-nv (the box test module redirects SPLIT_NV_DIR to a temp dir).
Run on the box host with the box's venv (torch for split_nv.engine):
    CUDA_VISIBLE_DEVICES= /home/ian/.venv/bin/python og_serve/interop_box.py /home/ian/split-nv-dsbuild
Every case checks the output equals the toy LM's greedy text and every verified step's rows equal the box's STEP
function of (keep, ids).
"""
import array
import importlib.util
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

BOX = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/ian/split-nv-dsbuild').resolve()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# The box test's fake prefill has no prefix-cache snapshot keys: rebuild sessions without OPEN "cache" here.
os.environ['DS41_OG_REOPEN_CACHE'] = '0'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_stepd as S  # noqa: E402  (pipe_wire, stepd, MacSim of this tree)

spec = importlib.util.spec_from_file_location('box_stepd_test', BOX / 'tools/test_dspark_stepd.py')
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)  # module level only: env, imports of split_nv (front, engine, dspark_wire)
BW = T.W  # the box's dspark_wire


class BoxWorld:
    """The box test's encodings: taps/keys carry (token, position); rows are the fake StepRunner's function."""
    port = None
    identity = 't'

    @staticmethod
    def tap(truth, pos):
        return np.frombuffer(T.enc_row(truth.seq[pos], pos, BW.TAP_ROW_BYTES), '<u2').copy()

    @staticmethod
    def key(truth, stage, pos):
        return np.frombuffer(T.enc_row(truth.seq[pos], pos, BW.KEY_ROW_BYTES), '<u2').copy()

    @staticmethod
    def rows_ok(truth, raw, keep, ids):
        return bytes(raw) == T.expected_payload(truth.seq, keep, ids)

    @staticmethod
    def open(enc, truth):
        # The fake box answers an OPEN with the ACK only (state "none"): open through reopen() at N-1.
        enc.history = array.array('I', truth.seq[:truth.N - 1])
        enc.reopen(truth.N - 1, truth.seq[truth.N - 1])


def toy_truth(prompt_len, gen_len, seed):
    prompt = T.prompt_of(prompt_len, seed=seed)
    return S.Truth(prompt_len, gen_len, seed, seq=prompt + T.greedy(prompt, gen_len + 16))


def sim(f, n, gen, seed, **kw):
    BoxWorld.port = f.srv.getsockname()[1]
    return S.MacSim(toy_truth(n, gen, seed), world=BoxWorld, max_tokens=gen, **kw)


def box_stats(f):
    return {k: v for k, v in f.dspark_summary().items() if k in ('cycles', 'fallbacks', 'rings', 'step_taps', 'hard_errors')}


def fresh():
    f = T.make_front(nslots=4)
    f.srv = T.serve(f)
    T.set_flags({})
    S.STATS.clear()
    S.stepd.bind(S.STATS)
    return f


def case_box_drafting():
    f = fresh()
    m = sim(f, 1500, 300, 3).run()
    m.check(None)
    st = box_stats(f)
    assert st['cycles']['box'] == S.STATS['box_draft_cycles'] > 50 and not st['hard_errors'], (st, S.counts())
    rows = sum(r for _, r in m.verifies) / len(m.verifies)
    assert rows > 2.5, rows  # the box ring decodes to the right tokens: its drafts are mostly right
    return f"{st['cycles']['box']} box cycles, {rows:.2f} rows/verify, RINGs {st['rings']}"


def case_crossing():
    f = fresh()
    m = sim(f, 800, 400, 5).run()
    m.check(None)
    st = box_stats(f)
    assert st['cycles']['explicit'] > 0 and st['cycles']['box'] > 0 and st['rings'] == 1, st
    return f"plain -> RING -> mode 1 x{st['cycles']['explicit']} -> mode 0 x{st['cycles']['box']}"


def case_kill_switch_fillers():
    f = fresh()
    m = sim(f, 1300, 400, 7)

    def on_cycle(i, keep):
        if i == 10:
            T.set_flags({'dspark': False})
        if i == 35:
            T.set_flags({})
    m.on_cycle = on_cycle
    m.run()
    m.check(None)
    st = box_stats(f)
    assert st['cycles']['filler'] >= 1 and S.STATS['box_draft_fillers'] == st['cycles']['filler'], (st, S.counts())
    assert not S.STATS.get('box_draft_drained') and st['cycles']['box'] > 0 and st['rings'] == 2, (st, S.counts())
    return f"{st['cycles']['filler']} filler(s) verified, Mac drafted while off, re-primed, box drafting again"


def case_step_taps():
    f = fresh()
    m = sim(f, 900, 300, 9, one_row_at=(3, 6)).run()
    m.check(None)
    st = box_stats(f)
    assert st.get('step_taps') == 2 and st['rings'] == 1 and st['cycles']['box'] > 0, st
    return 'Mac 1-row blocks as STEP + taps (V2.2): 1 RING, the ring stayed current'


def case_link_drop():
    f = fresh()
    m = sim(f, 1300, 400, 11)

    def on_cycle(i, keep):
        if i == 15:
            m.enc.sock.shutdown(socket.SHUT_RDWR)  # the link drops with a box-drafted STEPD in flight
    m.on_cycle = on_cycle
    m.run()
    m.check(None)
    st = box_stats(f)
    assert S.STATS['box_draft_ring_resends'] == 1 and st['rings'] == 2 and not st['hard_errors'], (st, S.counts())
    return 'reopen + RING + kickoff mode 0 on the real front, identical continuation'


def case_c2():
    f = fresh()
    a, b = sim(f, 1400, 250, 13, sessions=2), sim(f, 2100, 250, 15, sessions=2)
    live = [a.cycles_(), b.cycles_()]
    while live:
        for g in list(live):
            try:
                next(g)
            except StopIteration:
                live.remove(g)
    a.check(None)
    b.check(None)
    st = box_stats(f)
    assert st['cycles']['box'] > 100 and not st['hard_errors'], st
    return f"2 sessions interleaved on 2 ring slots, {st['cycles']['box']} box cycles"


def case_alias():
    f = fresh()
    saved, S.wire.STEPD_TAG = S.wire.STEPD_TAG, b'STEP'
    try:
        m = sim(f, 1300, 150, 17).run()
    finally:
        S.wire.STEPD_TAG = saved
    m.check(None)
    assert box_stats(f)['cycles']['box'] > 0
    return 'STEPD under b"STEP" + 18-byte header (V2.1 alias) accepted'


def case_extra_source():
    """Tool prompts: extra-source / copy proposals as STEPD mode 1 per cycle between box-drafted cycles (one RING)."""
    f = fresh()
    m = sim(f, 1300, 400, 19, extra_every=3, copy_every=5).run()
    m.check(None)
    st = box_stats(f)
    extra = S.STATS['box_draft_fallback_extra_source']
    assert extra > 0 and st['cycles']['explicit'] >= extra + S.STATS.get('box_draft_fallback_copy', 0), (st, S.counts())
    assert st['cycles']['box'] > 0 and st['rings'] == 1 and not st['hard_errors'] and m.ctl.sticky is None, st
    return f"{extra} extra-source + {S.STATS.get('box_draft_fallback_copy', 0)} copy (mode 1), {st['cycles']['box']} box cycles, 1 RING"


CASES = [case_box_drafting, case_crossing, case_kill_switch_fillers, case_step_taps, case_link_drop, case_c2, case_alias,
         case_extra_source]


def main():
    passed = 0
    for fn in CASES:
        t0 = time.time()
        try:
            note = fn()
            passed += 1
            print(f'PASS {fn.__name__[5:]} ({time.time() - t0:.1f}s): {note}', flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f'FAIL {fn.__name__[5:]}: {exc!r}', flush=True)
    print(f'{passed}/{len(CASES)} passed', flush=True)
    print('ALL PASS' if passed == len(CASES) else 'SOME FAILED', flush=True)
    return 0 if passed == len(CASES) else 1


if __name__ == '__main__':
    raise SystemExit(main())
