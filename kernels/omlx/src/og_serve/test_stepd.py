"""CPU-only end-to-end tests of DSpark drafting on the box (stepd.py + pipe_wire RING/STEPD/STPD, STEPD-SPEC.md v1)
against the fake box (fakebox.py). No MLX, no GPU, no real box.

MacSim plays og_model + batch_generator on a fake greedy target whose continuation is `truth`, in og_model's order:
after each verify's accept the stepd hook runs (BoxDraft.prepare: STEPD mode 0 / copy ids as mode 1 / the Mac drafts);
a Mac-drafted block goes out as STEPD mode 1 (presend); before the next verify a box-drafted block is resolved
(BoxDraft.resolve, or the Mac drafts it and redraft() sends it); the verify's rows come from ensure/recv_step_safe.
Every case checks:
  - identity: the emitted tokens equal `truth` (drafting never changes outputs);
  - every verified step's rows are the box's rows for exactly the verified ids after exactly the committed tokens;
  - no verify has one row (L=1 is bitwise different from the served L>=2 path);
  - every box-drafted cycle saw a full, correct ring (tap/key contents checked by the fake box).

Run: python og_serve/test_stepd.py   (PIPE_WIRE_BASE=<path of today's pipe_wire.py> for the flag-off byte test)
"""
import importlib.util
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
V41 = HERE.parent / 'omlx/patches/deepseek_v41'
sys.path.insert(0, str(HERE))
os.environ.setdefault('DS41_OG_RESUME_WAIT_S', '6')
os.environ.setdefault('DS41_OG_RESTART_WAIT_S', '6')
os.environ.setdefault('DS41_OG_STEP_TIMEOUT_S', '5')
os.environ.setdefault('DS41_OG_SPIN_MS', '0')
os.environ['DS41_OG_BOX_DRAFT'] = '1'


def load(name, path=None, register=True):
    spec = importlib.util.spec_from_file_location(name, str(path or V41 / f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dw = load('dspark_wire')
wire = load('pipe_wire')
stepd = load('stepd')
from fakebox import FakeBox, digest  # noqa: E402

PORT = int(os.environ.get('FAKEBOX_STEPD_PORT', '12630'))
PIPE = ((524288, (25.6, 27.8, 30.1, 32.8)), (131072, (24.9, 26.9, 29.0, 31.7)), (0, (24.1, 26.1, 28.2, 30.8)))
FUSED = ((524288, (13.7, 15.8, 17.9, 19.9)), (131072, (13.4, 15.3, 17.2, 19.2)), (0, (13.1, 15.0, 16.9, 18.8)))
VOCAB = 60000
STATS = {}
stepd.bind(STATS)


def tap(pos, token):
    return ((np.arange(stepd.TAP_DIM, dtype=np.uint32) * 3 + pos * 131 + token) % 65521).astype(np.uint16)


def key(stage, pos):
    return ((np.arange(stepd.KEY_DIM, dtype=np.uint32) * 5 + pos * 17 + stage * 7919) % 65521).astype(np.uint16)


def wrong(pos, token):
    return (token + 1 + pos % 7) % VOCAB


class Truth:
    def __init__(self, prompt_len, gen_len, seed, seq=None):
        rng = random.Random(seed)
        self.seq = seq or [rng.randrange(2, VOCAB) for _ in range(prompt_len + gen_len + 16)]
        self.N = prompt_len

    def draft(self, keep, width, salt):
        """A drafter that is right most of the time: truth with a deterministic error pattern."""
        out = []
        for i in range(width):
            p = keep + 1 + i
            t = self.seq[p]
            out.append(wrong(p, t) if (p * 7 + salt) % 9 == 0 else t)
        return out


def box_for(truth):
    """draft_fn / tap_ok / key_ok of a fake box serving this truth."""
    def draft_fn(history, keep, anchor):
        probs = [[0.9, 0.8, 0.7, 0.6], [0.9, 0.5, 0.3, 0.1], [0.6, 0.2, 0.1, 0.1]][keep % 3]
        return truth.draft(keep, 4, 3), probs

    def tap_ok(pos, raw):
        return raw == tap(pos, truth.seq[pos]).tobytes()

    def key_ok(stage, pos, raw):
        return raw == key(stage, pos).tobytes()
    return draft_fn, tap_ok, key_ok


class FakeWorld:
    """How the fake box encodes things: tap/key rows (checked by its tap_ok/key_ok) and history-digest rows."""
    port = PORT

    @staticmethod
    def tap(truth, pos):
        return tap(pos, truth.seq[pos])

    @staticmethod
    def key(truth, stage, pos):
        return key(stage, pos)

    @staticmethod
    def rows_ok(truth, raw, keep, ids):
        prefix = truth.seq[:keep]
        return all(raw[i * wire.ROW_BYTES:i * wire.ROW_BYTES + 32] == digest(prefix + ids[:i + 1])
                   for i in range(len(ids)))

    @staticmethod
    def open(enc, truth):
        enc.open(truth.seq[:truth.N])


class MacSim:
    """og_model + batch_generator's cycle on a fake target (see the module docstring)."""

    def __init__(self, truth, *, ask=True, sessions=1, copy_every=0, extra_every=0, max_tokens=300, wire_module=None,
                 one_row_at=(), world=FakeWorld):
        self.w, self.world = wire_module or wire, world
        self.truth, self.sessions, self.copy_every, self.max_tokens = truth, sessions, copy_every, max_tokens
        # extra_every: a tool-prompt extra source (draft_sources.Tracker) that proposes on these cycles; it is appended
        # every cycle with the committed ids and their taps (checked), whatever drafts the block
        self.extra_every, self.tracker = extra_every, [] if extra_every else None
        self.enc = self.w.EncoderSession('127.0.0.1', world.port, **({'identity': world.identity}
                                                                      if hasattr(world, 'identity') else {}))
        if ask and stepd.wants(type('P', (), dict(temperature=0.0, max_tokens=max_tokens))(), truth.N):
            self.enc.dspark_request = stepd.open_request(PIPE, FUSED)
        world.open(self.enc, truth)
        self.ctl = stepd.BoxDraft() if getattr(self.enc, 'dspark', None) is not None else None
        self.enc.stepd = self.ctl
        self.out, self.verifies, self.cycles = [], [], 0
        self.mac_end = None  # the Mac GPU ring holds positions up to here
        self.block = None
        self.actions = []
        self.rebuilds = 0
        self.on_cycle = None  # test hook, called with (cycle index, keep) before each verify
        self.one_row_at, self.one_row_keeps = set(one_row_at), set()  # Mac cycles that draft nothing (a 1-row block)

    # ---- the fake target and the verify ------------------------------------------------------------------------
    def target(self, keep, ids, i):
        seq = self.truth.seq
        p = keep + i + 1
        return seq[p] if ids[:i + 1] == seq[keep:keep + i + 1] else wrong(p, seq[p])

    def step(self, ids, keep, verify=True):
        ctl = self.ctl
        if ctl is not None and self.enc._pending is None and ctl.pending_explicit(keep):
            ctl.presend(self.enc, keep, ids)  # og_model._remote
        self.enc.ensure_step_safe(ids, keep)
        raw, _ = self.enc.recv_step_safe(ids, keep)
        assert self.world.rows_ok(self.truth, raw, keep, ids), f'box rows at {keep} do not match the committed tokens + {ids}'
        if verify:
            self.verifies.append((keep, len(ids)))
        return raw

    # ---- the stepd hook and the Mac drafter --------------------------------------------------------------------
    def mac_draft(self, keep, width=None):
        return self.truth.draft(keep, width or 1 + keep % 4, 5)

    def rebuild(self, keys, taps, end):
        k, t = keys.shape[1], taps.shape[0]
        for j in range(k):
            for s in range(stepd.STAGES):
                assert (keys[s, j] == self.world.key(self.truth, s, end - k - t + j)).all(), 'rebuilt Mac ring: wrong key'
        for j in range(t):
            p = end - t + j
            assert (taps[j] == self.world.tap(self.truth, p)).all(), 'rebuilt Mac ring: wrong tap row'
        self.mac_end = end
        self.rebuilds += 1

    def mac_keys(self):
        end = self.mac_end
        rows = np.arange(max(0, end - stepd.RING), end)
        return np.stack([np.stack([self.world.key(self.truth, s, p) for p in rows]) for s in range(stepd.STAGES)]), end

    def after_accept(self, base, n, anchor, verify):
        keep = base + n
        budget = self.max_tokens - len(self.out)
        self.cycles += 1
        action = 'plain'
        if self.ctl is not None:
            read = {}

            def taps_fn():
                read['taps'] = np.stack([self.world.tap(self.truth, p) for p in range(base, keep)])
                return read['taps']

            def propose_fn():  # og_model._box_prepare: the extra source first (appended every cycle), then the copy index
                if self.tracker is not None:
                    self.tracker.append((base, list(self.truth.seq[base + 1:keep + 1]), read['taps']))
                    if self.cycles % self.extra_every == 0:
                        return self.truth.draft(keep, 4, 2), 'schema'
                if self.copy_every and self.cycles % self.copy_every == 0:
                    return self.truth.draft(keep, 4, 1), 'copy'
                return None, None
            action, ids = self.ctl.prepare(
                self.enc, owner=self, base=base, n=n, anchor=anchor, verify=verify, taps_fn=taps_fn,
                mac_keys_fn=self.mac_keys, rebuild_fn=self.rebuild, propose_fn=propose_fn,
                sessions=self.sessions, fused=False, budget=budget)
        self.actions.append((keep, action))
        if action == 'box':
            self.block = ('box', None)
            return
        if action in ('copy', 'extra'):
            self.block = ('ids', ids)
            return
        if self.tracker is not None and base >= stepd.MIN_CTX:  # today's _dspark_prepare appends it in the Mac era
            self.tracker.append((base, list(self.truth.seq[base + 1:keep + 1]), None))
        # Today's Mac drafter: appends [base, keep) to its ring, then drafts (presend sends the block).
        assert self.mac_end == base, f'Mac drafts on a stale ring (ends {self.mac_end}, cycle base {base})'
        self.mac_end = keep
        drafts = self.mac_draft(keep)
        if self.cycles in self.one_row_at:
            drafts = []
            self.one_row_keeps.add(keep)
        self.block = ('ids', drafts)
        if verify is not None and self.enc._pending is None:  # og_model.presend (after the chain)
            if self.ctl is None or not self.ctl.presend(self.enc, keep, [anchor] + drafts):
                self.enc.send_step([anchor] + drafts, keep)

    def drafts_for(self, keep, anchor):
        kind, ids = self.block
        if kind == 'ids':
            return ids
        drafts = self.ctl.resolve(self.enc)  # og_model.mtp_box_resolve
        if drafts:
            return drafts
        self.ctl.ensure_mac_ring(self.rebuild)
        assert self.mac_end == keep
        drafts = self.mac_draft(keep, 4)
        if self.enc._pending is None:
            self.ctl.redraft(self.enc, keep, [anchor] + drafts)
        return drafts

    # ---- one request -------------------------------------------------------------------------------------------
    def run(self):
        for _ in self.cycles_():
            pass
        return self

    def cycles_(self):
        """One request, yielding after every verify (two requests interleave like c2 on the engine thread)."""
        seq, N = self.truth.seq, self.truth.N
        self.step([seq[N - 1]], N - 1, verify=False)  # Job kickoff (last prompt token)
        self.out.append(seq[N])  # the first token
        self.enc.send_step([seq[N]], N)  # mtp_first_presend; the post-init forward finds it
        self.step([seq[N]], N, verify=False)
        anchor = seq[N + 1]
        self.out.append(anchor)
        self.mac_end = N  # take_primed: the prompt ring
        self.after_accept(N, 1, anchor, None)
        keep = N + 1
        while True:
            if self.on_cycle is not None:
                self.on_cycle(len(self.verifies), keep)
            ids = [anchor] + self.drafts_for(keep, anchor)
            self.step(ids, keep)
            argmax = [self.target(keep, ids, i) for i in range(len(ids))]
            m = dw.accept_count(ids, argmax)
            committed = ids[1:m + 1] + [argmax[m]]
            room = self.max_tokens - len(self.out)
            if len(committed) >= room:
                self.out += committed[:room]
                break
            self.out += committed
            self.after_accept(keep, m + 1, committed[-1], (len(ids), m, argmax, ids))
            keep, anchor = keep + m + 1, committed[-1]
            yield keep
        self.enc.close()

    def check(self, box, *, drafted=True):
        want = self.truth.seq[self.truth.N:self.truth.N + self.max_tokens]
        assert self.out == want, f'output differs at {next(i for i, (a, b) in enumerate(zip(self.out, want)) if a != b)}'
        assert all(rows >= 2 or keep in self.one_row_keeps for keep, rows in self.verifies), 'a one-row verify'
        if box is None:  # not the fake box (og_serve/interop_box.py): no box log to read
            return None
        drafts = [e for e in box.log if e[0] == 'draft']
        bad = [e for e in drafts if e[3] != e[4]]
        assert not bad, f'box drafted on an incomplete/wrong ring: {bad[:3]}'
        if drafted:
            assert drafts, 'the box never drafted'
        early = [e for e in box.log if e[0] == 'stepd' and e[5] == dw.MODE_BOX and e[2] < 1024]
        assert not early, f'mode 0 below 1024: {early[:3]}'
        return drafts


def fresh_box(**kw):
    box = FakeBox(PORT, dspark=kw.pop('dspark', 'on'), **kw)
    return box


def run_case(truth, box, **kw):
    draft_fn, tap_ok, key_ok = box_for(truth)
    box.draft_fn, box.tap_ok, box.key_ok = draft_fn, tap_ok, key_ok
    return MacSim(truth, **kw)


def counts():
    return {k[len('box_draft_'):]: v for k, v in STATS.items() if v}


# ---- cases -----------------------------------------------------------------------------------------------------
def case_box_cycles(box):
    STATS.clear(); stepd.bind(STATS)
    sim = run_case(Truth(1500, 400, 1), box, max_tokens=400).run()
    drafts = sim.check(box)
    kinds = {a for _, a in sim.actions}
    assert kinds == {'box'}, kinds
    assert STATS['box_draft_cycles'] == len(drafts) == len(sim.verifies), (STATS['box_draft_cycles'], len(drafts))
    assert STATS['box_draft_rings'] == 1 and STATS['box_draft_kickoffs'] == 1
    assert sum(r for _, r in sim.verifies) - len(sim.verifies) == STATS['box_draft_drafted_tokens']
    return f'{len(drafts)} box cycles, {len(sim.out)} tokens, {sum(r for _, r in sim.verifies)/len(sim.verifies):.2f} rows/verify'


def case_crossing_1024(box):
    STATS.clear(); stepd.bind(STATS)
    sim = run_case(Truth(800, 500, 2), box, max_tokens=500).run()
    sim.check(box)
    first_ring = next(e for e in box.log if e[0] == 'ring')
    first_stepd = next(e for e in box.log if e[0] == 'stepd')
    assert first_stepd[2] >= stepd.PRIME_CTX > first_ring[2] - 1 and first_stepd[2] - first_stepd[8] == first_ring[2], \
        (first_ring, first_stepd)  # RING covers [.., base); the first STEPD carries [base, keep >= 896)
    plain = [k for k, a in sim.actions if a == 'plain']
    mac = [k for k, a in sim.actions if a == 'mac']
    box_ = [k for k, a in sim.actions if a == 'box']
    assert plain and max(plain) < stepd.PRIME_CTX + 5, plain[-3:]
    assert mac and min(mac) >= stepd.PRIME_CTX and max(mac) <= stepd.MIN_CTX + 5, (mac[:2], mac[-2:])
    assert box_ and min(box_) > stepd.MIN_CTX, box_[:2]
    steps = [e for e in box.log if e[0] in ('step', 'stepd') and e[2] >= stepd.PRIME_CTX]
    assert all(e[0] == 'stepd' for e in steps if e[2] > first_ring[2]), 'a plain STEP after priming'
    return f'plain until {max(plain)}, RING at {first_ring[2]}, mode 1 {min(mac)}..{max(mac)}, mode 0 from {min(box_)}'


def case_short_no_ask(box):
    STATS.clear(); stepd.bind(STATS)
    sim = run_case(Truth(200, 100, 3), box, max_tokens=100).run()
    sim.check(box, drafted=False)
    assert not any(e[0] == 'open' and e[4] for e in box.log), 'a short request asked for the grant'
    assert not any(e[0] in ('stepd', 'ring') for e in box.log)
    return 'no OPEN.dspark, plain STEPs only'


def case_no_grant(box):
    STATS.clear(); stepd.bind(STATS)
    box.dspark = 'not_loaded'
    sim = run_case(Truth(1500, 200, 4), box, max_tokens=200).run()
    sim.check(box, drafted=False)
    assert sim.ctl is None and sim.enc.dspark_off == 'not_loaded'
    assert not any(e[0] in ('stepd', 'ring') for e in box.log)
    return 'dspark_off not_loaded -> plain path'


def case_box_restart(box):
    STATS.clear(); stepd.bind(STATS)
    truth = Truth(1300, 500, 5)
    sim = run_case(truth, box, max_tokens=500)
    hits = []

    def on_cycle(i, keep):
        if i in (20, 45):  # a mode 0 STEPD is in flight: the engine dies before replying
            box.kill()
            hits.append(keep)
    sim.on_cycle = on_cycle
    sim.run()
    sim.check(box)
    assert len(hits) == 2 and STATS['box_draft_ring_resends'] == 2, (hits, counts())
    rings = [e for e in box.log if e[0] == 'ring']
    assert len(rings) == 3 and all(e[3] + e[4] == 128 for e in rings), rings
    assert STATS['box_draft_recoveries'] == 2
    return f'killed at {hits}, 2 RING resends (keys {rings[1][3]} + taps {rings[1][4]}), continuation identical'


def case_box_down_up(box):
    STATS.clear(); stepd.bind(STATS)
    sim = run_case(Truth(1300, 300, 6), box, max_tokens=300)

    def on_cycle(i, keep):
        if i == 15:
            box.down()
            threading.Timer(1.5, box.up).start()
    sim.on_cycle = on_cycle
    t0 = time.time()
    sim.run()
    sim.check(box)
    assert STATS['box_draft_ring_resends'] == 1, counts()
    return f'engine restart ridden out in {time.time() - t0:.1f}s, ring re-primed'


def case_kill_switch(box):
    STATS.clear(); stepd.bind(STATS)
    sim = run_case(Truth(1300, 500, 7), box, max_tokens=500)

    def on_cycle(i, keep):
        if i == 10:
            box.kill_switch = True
        if i == 40:
            box.kill_switch = False
    sim.on_cycle = on_cycle
    sim.run()
    sim.check(box)
    fillers = [e for e in box.log if e[0] == 'filler']
    assert fillers and all(e[3] == 1 for e in fillers) and not any(e[0] == 'bonus' for e in box.log), fillers
    notaps = [e for e in box.log if e[0] == 'stepd' and e[3] and e[8] == 0 and e[5] == dw.MODE_EXPLICIT]
    assert notaps, 'no NO_TAPS STEPD while the box was off'
    rings = [e for e in box.log if e[0] == 'ring']
    after = [k for k, a in sim.actions if a == 'box']
    assert len(rings) == 2 and max(after) > rings[1][2], (rings, after[-3:])
    assert rings[1][3:5] == (128, 0), f'the re-prime after the kill switch sends the Mac ring keys: {rings[1]}'
    assert STATS['box_draft_fillers'] == len(fillers) and not STATS.get('box_draft_drained') \
        and not STATS.get('box_draft_redrafts'), counts()
    assert STATS['box_draft_cycles'] + STATS['box_draft_fillers'] == sum(1 for _, a in sim.actions if a == 'box')
    return f'{len(fillers)} V2.3 filler replies verified as 1-draft blocks, {len(notaps)} NO_TAPS steps, re-primed at {rings[1][2]}'


def case_pf_active(box):
    STATS.clear(); stepd.bind(STATS)
    sim = run_case(Truth(1300, 500, 8), box, sessions=2, max_tokens=500)
    flips = []

    def on_cycle(i, keep):
        if i == 10:
            box.pf_active = True
            flips.append(keep)
        if i == 30:
            box.pf_active = False
            flips.append(keep)
    sim.on_cycle = on_cycle
    sim.run()
    sim.check(box)
    mac = [k for k, a in sim.actions if a == 'mac']
    assert mac and flips[0] <= min(mac) and max(mac) <= flips[1] + 40, (flips, mac[:2], mac[-2:])
    assert STATS['box_draft_fallback_pf_active'] == len(mac)
    rings = [e for e in box.log if e[0] == 'ring']
    during = [e for e in box.log if e[0] == 'stepd' and e[5] == dw.MODE_EXPLICIT and min(mac) < e[2] <= max(mac)]
    if stepd.PF_TAPS:
        assert len(rings) == 1, 'the ring had to be re-primed after the Mac drafted with mode 1 + taps'
    else:  # NO_TAPS while PF_ACTIVE (a plain step on the box), then a keys-only re-prime from the Mac ring
        assert during and all(e[8] == 0 for e in during), during[:3]
        assert len(rings) == 2 and rings[1][3:5] == (128, 0) and rings[1][2] >= max(mac), rings
        box_after = [k for k, a in sim.actions if a == 'box' and k > max(mac)]
        assert box_after and min(box_after) > rings[1][2], (rings[1], box_after[:2])
    # c1 + a prefill keeps box drafting
    STATS.clear(); stepd.bind(STATS)
    box.pf_active = True
    solo = run_case(Truth(1300, 200, 9), box, sessions=1, max_tokens=200).run()
    box.pf_active = False
    solo.check(box)
    assert {a for _, a in solo.actions} == {'box'}
    how = 'ring kept by mode 1 + taps' if stepd.PF_TAPS else 'NO_TAPS, then a keys-only RING'
    return f'Mac drafted {len(mac)} cycles while PF_ACTIVE at c2 ({how}), c1 kept box drafting'


def case_pf_active_taps(box):
    """DS41_OG_BOX_PF_TAPS=1: the PR's first behaviour (mode 1 keeps the ring current with taps, no re-prime)."""
    saved, stepd.PF_TAPS = stepd.PF_TAPS, True
    try:
        return case_pf_active(box)
    finally:
        stepd.PF_TAPS = saved


def case_v1_box_bonus_only(box):
    """A v1 box (no filler, no step_taps): a mode 0 fallback is bonus only; the Mac drops it and redrafts."""
    STATS.clear(); stepd.bind(STATS)
    box.v2 = False
    sim = run_case(Truth(1300, 500, 21), box, max_tokens=500)

    def on_cycle(i, keep):
        if i == 10:
            box.kill_switch = True
        if i == 40:
            box.kill_switch = False
    sim.on_cycle = on_cycle
    sim.run()
    sim.check(box)
    bonus = [e for e in box.log if e[0] == 'bonus']
    assert bonus and all(e[3] == 1 for e in bonus) and not any(e[0] == 'filler' for e in box.log), bonus
    assert 'step_taps' not in sim.enc.dspark and 'filler' not in sim.enc.dspark
    assert STATS['box_draft_drained'] == len(bonus) and STATS['box_draft_redrafts'] == len(bonus)
    return f'{len(bonus)} bonus-only replies dropped + redrafted (kickoff mode 1), identical'


def case_step_taps(box):
    """A Mac block STEPD mode 1 cannot carry (one row) goes out as a plain STEP with the taps (V2.2): no RING resend."""
    STATS.clear(); stepd.bind(STATS)
    sim = run_case(Truth(900, 400, 22), box, max_tokens=400, one_row_at=(3, 6, 9)).run()
    sim.check(box)
    taps = [e for e in box.log if e[0] == 'step_taps']
    assert len(taps) == 3 and all(e[5] and e[3] == 1 for e in taps), taps
    assert len([e for e in box.log if e[0] == 'ring']) == 1 and STATS['box_draft_step_taps'] == 3
    assert any(a == 'box' for _, a in sim.actions)
    return f'3 plain STEPs with taps appended at {[e[2] for e in taps]}, 1 RING, box drafting after'


def case_stepd_alias(box):
    """V2.1: the box also takes a STEPD sent as b"STEP" with the 18-byte header (b0b1a444's reading)."""
    STATS.clear(); stepd.bind(STATS)
    saved, wire.STEPD_TAG = wire.STEPD_TAG, b'STEP'
    try:
        sim = run_case(Truth(1300, 200, 23), box, max_tokens=200).run()
    finally:
        wire.STEPD_TAG = saved
    sim.check(box)
    assert any(e[0] == 'stepd_alias' for e in box.log)
    return 'STEPD under b"STEP" + 18-byte header accepted, identical'


def case_accept_mismatch(box):
    STATS.clear(); stepd.bind(STATS)
    sim = run_case(Truth(1300, 300, 10), box, max_tokens=300)

    def on_cycle(i, keep):
        if i == 12:
            box.err_stepd = 'accept_mismatch'
    sim.on_cycle = on_cycle
    sim.run()
    sim.check(box)
    assert sim.ctl.sticky == 'box_error' and STATS['box_draft_errors'] >= 1, (sim.ctl.sticky, counts())
    tail = [a for _, a in sim.actions[-20:]]
    assert set(tail) == {'plain'}, tail
    return 'hard ERR -> session rebuilt, rest of the request on the Mac drafter, identical'


def case_hard_error_in_mode1(box):
    STATS.clear(); stepd.bind(STATS)
    sim = run_case(Truth(800, 400, 11), box, max_tokens=400)

    def on_cycle(i, keep):
        if keep > 950 and box.err_stepd is None and not getattr(box, '_fired', False):
            box.err_stepd, box._fired = 'bad_stepd', True
    sim.on_cycle = on_cycle
    sim.run()
    sim.check(box, drafted=False)
    assert sim.ctl.sticky == 'box_error'
    return 'hard ERR on a mode 1 STEPD -> recover resends the plain STEP, identical'


def case_cancellation(box):
    STATS.clear(); stepd.bind(STATS)
    truth = Truth(1300, 300, 12)
    sim = run_case(truth, box, max_tokens=300)
    box.step_delay = 0.3
    done = []

    def on_cycle(i, keep):
        if i == 5:
            sim.enc.close()  # the request is cancelled while its box-drafted STEPD is in flight
            done.append(keep)
            raise KeyboardInterrupt
    sim.on_cycle = on_cycle
    try:
        sim.run()
    except KeyboardInterrupt:
        pass
    time.sleep(0.5)
    box.step_delay = 0.0
    assert done and any(e[0] == 'close' for e in box.log)
    other = run_case(Truth(1300, 150, 13), box, max_tokens=150).run()
    other.check(box)
    return 'CLOS with a STEPD in flight; the next request is unaffected'


def case_ring_rejected(box):
    STATS.clear(); stepd.bind(STATS)
    box.ring_reject = True
    sim = run_case(Truth(1300, 300, 14), box, max_tokens=300).run()
    sim.check(box)
    rings = [e for e in box.log if e[0] == 'ring']
    fillers = [e for e in box.log if e[0] == 'filler']
    assert len(rings) == 2 and len(fillers) == 1 and fillers[0][3] == 5, (rings, fillers)
    return 'RING sha mismatch -> filler (ring_rejected) verified, re-primed next cycle'


def case_copy(box):
    STATS.clear(); stepd.bind(STATS)
    sim = run_case(Truth(1300, 400, 15), box, copy_every=3, max_tokens=400).run()
    sim.check(box)
    copies = [k for k, a in sim.actions if a == 'copy']
    assert copies and STATS['box_draft_fallback_copy'] == len(copies)
    assert len([e for e in box.log if e[0] == 'ring']) == 1, 'copy cycles broke the ring'
    return f'{len(copies)} copy cycles as STEPD mode 1, ring stayed current'


def case_extra_source(box):
    """Tool prompts (DS41_EXTRA_DRAFT): extra-source proposals pre-empt the box per cycle as STEPD mode 1 (taps attached,
    the ring stays current); every other cycle is box-drafted; the tracker sees every committed row once, in order."""
    STATS.clear(); stepd.bind(STATS)
    truth = Truth(1300, 500, 24)
    sim = run_case(truth, box, extra_every=4, copy_every=5, max_tokens=500).run()
    sim.check(box)
    kinds = [a for k, a in sim.actions if k > stepd.MIN_CTX + 8]
    assert {'box', 'extra', 'copy'} <= set(kinds), set(kinds)
    extra = [k for k, a in sim.actions if a == 'extra']
    assert STATS['box_draft_fallback_extra_source'] == len(extra) == sim.ctl.counts['extra'] > 0, counts()
    assert STATS['box_draft_fallback_copy'] == sim.ctl.counts['copy'] and not sim.ctl.sticky
    explicit = [e for e in box.log if e[0] == 'stepd' and e[5] == dw.MODE_EXPLICIT and e[2] > stepd.MIN_CTX + 8]
    assert len(explicit) == len(extra) + sim.ctl.counts['copy'] and all(e[8] for e in explicit), explicit[:3]
    assert len([e for e in box.log if e[0] == 'ring']) == 1, 'extra-source cycles broke the ring'
    rows = sim.tracker
    assert [b for b, _, _ in rows] == sorted(b for b, _, _ in rows), 'tracker appends out of order'
    for (b, ids, taps), (b2, _, _) in zip(rows, rows[1:]):
        assert b + len(ids) == b2, (b, len(ids), b2)  # contiguous: every committed row exactly once
        assert taps is None or [bytes(t) for t in taps] == [sim.world.tap(truth, p).tobytes() for p in range(b, b2)]
    summary = sim.ctl.summary()
    assert summary['extra'] == len(extra) and {'prep_ms', 'send_ms', 'rtt_ms', 'box_ms', 'wait_ms'} <= set(summary)
    return f'{len(extra)} extra-source + {sim.ctl.counts["copy"]} copy cycles as mode 1 among {sim.ctl.counts["box"]} box cycles, 1 RING'


def case_lost_grant(box):
    STATS.clear(); stepd.bind(STATS)
    sim = run_case(Truth(1300, 300, 16), box, max_tokens=300)

    def on_cycle(i, keep):
        if i == 12:
            box.dspark = 'not_loaded'  # the engine comes back without the drafter
            box.kill()
    sim.on_cycle = on_cycle
    sim.run()
    box.dspark = 'on'
    sim.check(box)
    assert sim.ctl.sticky == 'lost_grant', sim.ctl.sticky
    return 'reopen without the grant -> Mac drafter + plain STEP for the rest, identical'


def case_flag_off_bytes(box):
    """DS41_OG_BOX_DRAFT=0 / no grant: the Mac->box byte stream equals today's pipe_wire, byte for byte."""
    base = os.environ.get('PIPE_WIRE_BASE')
    if not base:
        try:
            src = subprocess.run(['git', '-C', str(HERE.parent), 'show', 'ds41-next4:omlx/patches/deepseek_v41/pipe_wire.py'],
                                 check=True, capture_output=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return 'SKIP (no base pipe_wire)'
        base = str(Path(os.environ.get('TMPDIR', '/tmp')) / 'pipe_wire_base_stepd.py')
        Path(base).write_bytes(src)
    old = load('pipe_wire_base', base, register=False)
    streams = []
    for module in (old, wire):
        frames = []
        real = module.send

        def send(sock, tag, header, payload=b'', real=real, frames=frames):
            h = header if isinstance(header, bytes) else module.json.dumps(header, separators=(',', ':')).encode()
            frames.append(module.FRAME.pack(tag, len(h), len(payload)) + h + bytes(payload))
            return real(sock, tag, header, payload)
        module.send = send
        box.log.clear()
        box.next_sid = 1
        truth = Truth(1200, 120, 17)
        box.draft_fn = box_for(truth)[0]
        try:
            sim = MacSim(truth, ask=False, max_tokens=120, wire_module=module)

            def on_cycle(i, keep):
                if i == 7:
                    box.kill()  # a recover in the stream too
                    time.sleep(0.2)
            sim.on_cycle = on_cycle
            sim.run()
            sim.check(box, drafted=False)
        finally:
            module.send = real
        streams.append(frames)
    assert len(streams[0]) == len(streams[1]), [len(x) for x in streams]
    assert streams[0] == streams[1], 'the flag-off byte stream differs from ds41-next4'
    tags = sorted({f[:4] for f in streams[0]})
    return f'{len(streams[0])} frames {tags}, {sum(map(len, streams[0]))} bytes identical to ds41-next4 pipe_wire'


def case_spec_vectors(box):
    """STEPD-SPEC.md §15 test vectors through the Mac's codec and grant/decision rules."""
    costs = dw.parse_costs(stepd.open_request(PIPE, FUSED)['costs'])
    probs = [0.9, 0.8, 0.5, 0.25]
    assert dw.box_depth(probs, costs, 69999, False, 4) == 3 and dw.box_depth(probs, costs, 69999, True, 4) == 3
    assert dw.box_depth(probs, costs, 69999, False, 2) == 2
    assert dw.accept_count([42, 500, 777], [500, 1234, 9]) == 1
    hdr, payload = dw.encode_stepd(7, 69999, 1234, 3, 1, 0, 4, 0, [500, 1234, 9], (), b'\0' * 2 * 30720)
    assert hdr.hex() == '070000006f110100d2040000030100040000' and len(payload) == 61452
    assert payload[:12].hex() == 'f4010000d204000009000000'
    hdr, payload = dw.encode_stepd(7, 69999, 1234, 0, 0, 1, 4, dw.F_FUSED, (), [11, 22])
    assert hdr.hex() == '070000006f110100d2040000000001040102' and payload.hex() == '0b00000016000000'
    stpd = dw.encode_stpd_header(7, 69999, 4, 0.00912, 1, 0, dw.R_DRAFTED | dw.R_RING_OK, 0, [11, 22, 33, 44],
                                 probs, 1.5)
    assert stpd.hex() == ('07000000731101000400d08502000d6c153c01030005000400000000c03f0b000000160000002100'
                          '00002c0000006666663fcdcc4c3f0000003f0000803e')
    import hashlib
    _, ring = dw.encode_ring(7, 69999, b'\0' * 3 * 2 * 1024, b'\0' * 30720, 2, 1)
    assert len(ring) == 36864 and hashlib.sha256(ring).hexdigest() == \
        '1c0273095382988333e2f2b5ae487cea460737ed9be65cbad9c5de537f95bf75'
    assert wire.grant_ok(dw.capability()) and not wire.grant_ok(dict(dw.capability(), ring=64))
    assert wire.STEPD_TAG == dw.TAG_STEPD == b'DSTP' and dw.TAG_STPD == b'STPD' and len(hdr) == 18
    assert dw.capability()['step_taps'] == 1 and dw.capability()['filler'] == 1
    r = stepd.HostRing()
    r.reset(np.stack([np.stack([key(s, p) for p in range(900, 1000)]) for s in range(3)]), 1000)
    for p in range(1000, 1100):
        r.append(tap(p, p)[None], p)
    keys, taps = r.window()
    assert keys.shape == (3, 28, 512) and taps.shape == (100, 15360)
    assert (keys[1, 0] == key(1, 972)).all() and (taps[-1] == tap(1099, 1099)).all()
    for p in range(1100, 1300):
        r.append(tap(p, p)[None], p)
    keys, taps = r.window()
    assert keys.shape[1] == 0 and taps.shape[0] == 128 and (taps[0] == tap(1172, 1172)).all()
    return 'codec vectors, grant check, host ring windows'


def case_kickoff_mode1(box):
    """A prompt between 896 and 1024: the first block is a Mac-drafted kickoff (STEPD mode 1, nver 0, sent by _remote)."""
    STATS.clear(); stepd.bind(STATS)
    sim = run_case(Truth(950, 300, 18), box, max_tokens=300).run()
    sim.check(box)
    first = next(e for e in box.log if e[0] == 'stepd')
    ring = next(e for e in box.log if e[0] == 'ring')
    assert first[3] == 0 and first[5] == dw.MODE_EXPLICIT and ring[2] == first[2] == 951, (ring, first)
    assert sim.actions[0] == (951, 'mac') and any(a == 'box' for _, a in sim.actions)
    return f'RING to {ring[2]} (keys {ring[3]} + taps {ring[4]}), kickoff mode 1, box drafting from 1024'


def case_c2_interleaved(box):
    """Two requests on one box, cycles alternating (c2): independent rings, both box-drafted, both identical."""
    STATS.clear(); stepd.bind(STATS)
    ta, tb = Truth(1400, 300, 19), Truth(2100, 300, 20)  # disjoint position ranges: the fake tells them apart
    fa, fb = box_for(ta), box_for(tb)
    pick = lambda pos: fa if pos < 2000 else fb  # noqa: E731
    box.draft_fn = lambda h, keep, anchor: pick(keep)[0](h, keep, anchor)
    box.tap_ok = lambda pos, raw: pick(pos)[1](pos, raw)
    box.key_ok = fa[2]
    a, b = MacSim(ta, sessions=2, max_tokens=300), MacSim(tb, sessions=2, max_tokens=300)
    live = [a.cycles_(), b.cycles_()]
    while live:
        for g in list(live):
            try:
                next(g)
            except StopIteration:
                live.remove(g)
    a.check(box)
    b.check(box)
    sids = {e[1] for e in box.log if e[0] == 'draft'}
    assert sids == {1, 2}, sids
    return f'2 sessions interleaved, {sum(1 for e in box.log if e[0] == "draft")} box cycles, both identical'


CASES = [case_spec_vectors, case_box_cycles, case_crossing_1024, case_kickoff_mode1, case_c2_interleaved, case_short_no_ask, case_no_grant, case_copy,
         case_extra_source,
         case_box_restart, case_box_down_up, case_kill_switch, case_v1_box_bonus_only, case_step_taps,
         case_stepd_alias, case_pf_active, case_pf_active_taps, case_ring_rejected,
         case_accept_mismatch, case_hard_error_in_mode1, case_lost_grant, case_cancellation, case_flag_off_bytes]


def main():
    passed = 0
    for fn in CASES:
        box = fresh_box()
        t0 = time.time()
        try:
            note = fn(box)
            passed += 1
            print(f'PASS {fn.__name__[5:]} ({time.time() - t0:.1f}s): {note}', flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f'FAIL {fn.__name__[5:]}: {exc!r}', flush=True)
        finally:
            box.down()
            time.sleep(0.2)
    print(f'{passed}/{len(CASES)} passed', flush=True)
    print('ALL PASS' if passed == len(CASES) else 'SOME FAILED', flush=True)
    return 0 if passed == len(CASES) else 1


if __name__ == '__main__':
    raise SystemExit(main())
