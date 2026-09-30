# SPDX-License-Identifier: MIT
"""ds41-og DSpark on the box (stepd.py, STEPD-SPEC.md v1): the Mac-side glue, CPU only.

The wire/lifecycle end to end (RING/STEPD/STPD, restarts, kill switch, PF_ACTIVE, errors, cancellation) runs against
the fake box in og_serve/test_stepd.py. These tests cover og_model / batch_generator / fused_batch: the hook order,
state updates, ring export and rebuild, and that nothing changes with DS41_OG_BOX_DRAFT=0.
"""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

import mlx.core as mx

from omlx.patches.deepseek_v41 import dspark_wire as dw
from omlx.patches.deepseek_v41 import og_model, pipe_wire, stepd
from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_lm_mtp import fused_batch
from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import DSparkContextCache


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture
def stats(monkeypatch):
    table = {}
    monkeypatch.setattr(stepd, "STATS", table)
    stepd.bind(table)
    return table


class FakeHost:
    """The DSpark host surface the ring code uses: 3 stage rings; the "key" of a tap row is a fixed projection."""

    def make_mtp_cache(self):
        return [DSparkContextCache(128) for _ in range(stepd.STAGES)]

    def dspark_append_context(self, main_hidden, cache, *, start_offset=None):
        for stage, item in enumerate(cache):
            kv = (main_hidden[..., :stepd.KEY_DIM] + stage).astype(mx.bfloat16)
            item.append(kv[:, None], start_offset=start_offset)


def _taps(start, n):
    rows = (np.arange(n)[:, None] * 7 + np.arange(stepd.TAP_DIM)[None] + start) % 251
    return mx.array(rows[None].astype(np.float32)).astype(mx.bfloat16)


class FakeEncoder:
    """Records RING / STEPD instead of sending them."""

    def __init__(self, grant=True):
        self.dspark = dw.capability() if grant else None
        self.session, self._pending, self._ready, self.sent = 1, None, None, []
        self.stepd = None

    def send_ring(self, offset, keys, taps, kr, tr):
        self.sent.append(("RING", offset, kr, tr, len(keys) + len(taps)))
        return len(keys) + len(taps)

    def send_stepd(self, keep, anchor, **kw):
        self.sent.append(("STEPD", keep, anchor, kw))
        self._pending = (0, keep, 0.0, None)

    def send_step(self, ids, keep, taps=b""):
        self.sent.append(("STEP", keep, list(ids), len(taps)))
        self._pending = (len(ids), keep, 0.0, tuple(ids))


def test_flag_off_is_inert(monkeypatch, stats):
    monkeypatch.setattr(stepd, "ENABLED", False)
    lm = og_model.OgLanguageModel.__new__(og_model.OgLanguageModel)
    assert lm.mtp_box_prepare(None, None, None, None) is None
    assert not stepd.wants(SimpleNamespace(temperature=0.0, max_tokens=4096), 8192)
    assert stats["box_draft_enabled"] == 0
    monkeypatch.setattr(og_model.og_fused, "ENABLED", True)
    monkeypatch.setattr(og_model, "FUSE_MIN", 3)
    assert og_model.OgLanguageModel.mtp_verify_groups(object(), [5, 5]) is None
    encoder = pipe_wire.EncoderSession.__new__(pipe_wire.EncoderSession)
    encoder.dspark_request = None
    encoder._grant({"dspark": dw.capability()})  # a grant nobody asked for is ignored
    assert getattr(encoder, "dspark", None) is None


def test_wants_and_open_request(monkeypatch):
    monkeypatch.setattr(stepd, "ENABLED", True)
    greedy = SimpleNamespace(temperature=0.0, max_tokens=512, repetition_penalty=1.0, presence_penalty=0.0,
                             frequency_penalty=0.0, thinking_budget=None, compiled_grammar=None)
    assert stepd.wants(greedy, 600)
    assert not stepd.wants(greedy, 400)  # cannot reach 1024
    assert not stepd.wants(SimpleNamespace(**{**vars(greedy), "temperature": 0.7}), 4096)
    assert not stepd.wants(SimpleNamespace(**{**vars(greedy), "thinking_budget": 100}), 4096)
    request = og_model.dspark_request()
    assert request["ver"] == 1 and dw.parse_costs(request["costs"])["pipe"][-1][0] == 0


def test_keys_export_and_rebuild_round_trip():
    """Mac ring -> host keys snapshot + taps -> rebuilt ring == the ring the Mac would have appended itself."""
    host = FakeHost()
    reference = host.make_mtp_cache()
    for start in range(0, 300, 60):  # the prompt ring (capture_prompt appends chunks)
        host.dspark_append_context(_taps(start, 60), reference, start_offset=start)
    keys, end = og_model._box_keys(reference)
    assert keys.shape == (3, 128, 512) and end == 300
    ring = stepd.HostRing()
    ring.reset(keys, end)
    for start in range(300, 390, 3):  # box-drafted cycles: taps only
        ring.append(np.array(_taps(start, 3).reshape(-1, stepd.TAP_DIM).view(mx.uint16)), start)
        host.dspark_append_context(_taps(start, 3), reference)
    rebuilt = og_model._box_rebuild(host, *ring.window(), ring.offset)
    for a, b in zip(reference, rebuilt):
        assert a.offset == b.offset == 390
        assert np.array_equal(np.array(a.keys.view(mx.uint16)), np.array(b.keys.view(mx.uint16)))
    for start in range(390, 600, 5):  # past 128 taps: no keys left in the window
        ring.append(np.array(_taps(start, 5).reshape(-1, stepd.TAP_DIM).view(mx.uint16)), start)
        host.dspark_append_context(_taps(start, 5), reference)
    rebuilt = og_model._box_rebuild(host, *ring.window(), ring.offset)
    for a, b in zip(reference, rebuilt):
        assert np.array_equal(np.array(a.keys.view(mx.uint16)), np.array(b.keys.view(mx.uint16)))


def _setup(monkeypatch, *, base, n=3, copy=None, greedy=True, sessions=1):
    monkeypatch.setattr(stepd, "ENABLED", True)
    host = FakeHost()
    prime = host.make_mtp_cache()
    host.dspark_append_context(_taps(base - 200, 200), prime, start_offset=base - 200)
    controller = SimpleNamespace(cost_policy=True, max_depth=4, cur=4)
    state = SimpleNamespace(hist_offset=base, next_main=mx.array([11], dtype=mx.uint32),
                            drafts=mx.array([21, 22, 23], dtype=mx.uint32), queue=deque([0] * n),
                            copy_index=copy, controller=controller, mtp_cache=prime,
                            last_verify=(n - 1, [21, 22, 99, 5], [21, 22, 23], mx.array([11], dtype=mx.uint32)))
    batch = SimpleNamespace(samplers=[SimpleNamespace(temp=0.0 if greedy else 0.7)], fallback_sampler=None,
                            logits_processors=[], max_tokens=[5000], _num_tokens=[100])
    encoder = FakeEncoder()
    encoder.stepd = stepd.BoxDraft()
    monkeypatch.setattr(og_model, "SESSIONS", {i: object() for i in range(sessions)})
    committed = mx.array([21, 22, 99], dtype=mx.uint32)[:n]
    hidden = _taps(base, n)
    return host, state, batch, encoder, committed, hidden


def test_box_prepare_box_era_sends_ring_then_stepd(monkeypatch, stats):
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000)
    handled = og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd)
    assert handled is True and state.box_pending and state.drafts is None and state.draft_source == "box"
    assert state.hist_offset == 5003 and "last_verify" not in vars(state)
    ring, step = encoder.sent
    assert ring[:4] == ("RING", 5000, 128, 0)  # the Mac's prime keys, up to base
    kind, keep, anchor, kw = step
    assert (kind, keep, anchor) == ("STEPD", 5003, 99)
    assert kw["nver"] == 4 and kw["a"] == 2 and kw["argmax"] == [21, 22, 99, 5] and kw["mode"] == dw.MODE_BOX
    assert kw["taps"] == np.array(hidden.reshape(-1, stepd.TAP_DIM).view(mx.uint16)).tobytes()
    assert kw["dmax"] == 4 and encoder.stepd.box["keep"] == 5003
    assert not encoder.stepd.mac_current and stats["box_draft_mode0"] == 1


def test_box_prepare_copy_first(monkeypatch, stats):
    copy = SimpleNamespace(appended=[], append=lambda ids: copy.appended.append(ids), propose=lambda budget: [7, 8])
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000, copy=copy)
    assert og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd) is True
    assert copy.appended == [[21, 22, 99]] and state.drafts.tolist() == [7, 8] and state.draft_source == "copy"
    assert encoder.sent[-1][3]["mode"] == dw.MODE_EXPLICIT and encoder.sent[-1][3]["explicit"] == [7, 8]
    assert not getattr(state, "box_pending", None)


def test_box_prepare_mac_era_below_1024_then_presend(monkeypatch, stats):
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=950)
    assert og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd) is None
    assert state.hist_offset == 950 and state.drafts.tolist() == [21, 22, 23]  # today's drafter runs next
    assert encoder.sent == [("RING", 950, 128, 0, 3 * 128 * 1024)]
    assert stats["box_draft_fallback_short_ctx"] == 1
    # og_model.presend: the Mac's block goes out as STEPD mode 1 with this cycle's taps
    state.drafts, state.next_main = mx.array([31, 32], dtype=mx.uint32), mx.array([99], dtype=mx.uint32)
    lm = og_model.OgLanguageModel.__new__(og_model.OgLanguageModel)
    gen_batch = SimpleNamespace(model=lm, prompt_cache=SimpleNamespace(size=lambda: 953))
    gen_batch.prompt_cache = [SimpleNamespace(size=lambda: 953)]
    monkeypatch.setattr(og_model, "session_of", lambda cache: encoder)
    og_model.presend(gen_batch, state)
    kind, keep, anchor, kw = encoder.sent[-1]
    assert (kind, keep, anchor, kw["mode"], kw["explicit"], kw["nver"]) == ("STEPD", 953, 99, 1, [31, 32], 4)


def test_box_prepare_not_capable_stays_on_today(monkeypatch, stats):
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000, greedy=False)
    assert og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd) is None
    assert encoder.sent == [] and encoder.stepd.sticky == "sampling" and stats["box_draft_fallback_sampling"] == 1


def test_box_prepare_pf_active_at_c2_rebuilds_the_mac_ring(monkeypatch, stats):
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000, sessions=2)
    ctl = encoder.stepd
    og_model._box_prepare(host, batch, state, hidden, committed, encoder, ctl)  # box era
    encoder._pending = None
    ctl.pf_hold = True  # a PF_ACTIVE reply; 2 sessions decode
    state.next_main = mx.array([7], dtype=mx.uint32)  # finish() already moved on (the draft_jobs order)
    committed2, hidden2 = mx.array([5, 7], dtype=mx.uint32), _taps(5003, 2)
    state.last_verify = (1, [5, 7, 8], [5, 6], mx.array([99], dtype=mx.uint32))  # verified [99, 5, 6]
    assert og_model._box_prepare(host, batch, state, hidden2, committed2, encoder, ctl) is None
    assert ctl.mac_current and stats["box_draft_mac_rebuilds"] == 1 and stats["box_draft_fallback_pf_active"] == 1
    assert [c.offset for c in state.mtp_cache] == [5003] * 3  # rebuilt through base; today's code appends [5003, 5005)


def test_clamped_commit_sends_no_stepd(monkeypatch, stats):
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000, n=2)
    state.last_verify = (1, [21, 22, 99, 5], [21, 22, 23], mx.array([11], dtype=mx.uint32))  # clamped: accept 2 -> 1
    assert og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd) is None
    assert encoder.sent == [] and stats["box_draft_fallback_clamp"] == 1


def test_presend_skips_a_box_pending_block(monkeypatch):
    state = SimpleNamespace(box_pending=True, next_main=mx.array([1], dtype=mx.uint32), drafts=None, queue=deque())
    og_model.presend(SimpleNamespace(model=None), state)  # returns before touching model/drafts


def test_verify_groups_c2_singletons_while_box_drafting(monkeypatch):
    monkeypatch.setattr(stepd, "ENABLED", True)
    monkeypatch.setattr(og_model.og_fused, "ENABLED", True)
    monkeypatch.setattr(og_model, "FUSE_MIN", 3)
    ctl = stepd.BoxDraft()
    enc = SimpleNamespace(stepd=ctl)
    monkeypatch.setattr(og_model, "SESSIONS", {1: enc, 2: SimpleNamespace(stepd=None)})
    groups = og_model.OgLanguageModel.mtp_verify_groups
    assert groups(object(), [5, 5]) is None  # the Mac ring is current: batched verify as today
    ctl.mac_current = False
    assert groups(object(), [5, 5]) == [[0], [1]]
    monkeypatch.setattr(og_model.og_fused, "_static_ok", lambda lm: True)
    monkeypatch.setattr(og_model.og_fused.mx, "default_device", lambda: og_model.og_fused.mx.gpu)
    assert groups(object(), [5, 5, 5]) == [[0, 1], [2]]  # c3+: fused pairs as today


def test_batch_generator_helpers():
    pending = SimpleNamespace(drafts=None, box_pending=True)
    plain = SimpleNamespace(drafts=mx.array([1, 2], dtype=mx.uint32))
    assert bg._box_pending(pending) and not bg._box_pending(plain)
    assert bg._verify_rows(pending) == 5 and bg._verify_rows(plain) == 3
    calls = []
    host = SimpleNamespace(_omlx_dspark_decode_enabled=True, mtp_box_resolve=lambda b, s: calls.append(s))
    batch = SimpleNamespace(model=host)
    bg._resolve_box_drafts(batch, plain)
    bg._resolve_box_drafts(batch, pending)
    assert calls == [pending]


def test_dspark_prepare_hook(monkeypatch):
    seen = []
    host = SimpleNamespace(_omlx_dspark_decode_enabled=True, args=SimpleNamespace(dspark_block_size=5),
                           mtp_box_prepare=lambda *a: seen.append(a) or True)
    copy = SimpleNamespace(append=lambda ids: pytest.fail("the copy index ran twice"))
    state = SimpleNamespace(controller=SimpleNamespace(cur=4), depth=4, copy_index=copy, hist_offset=5000)
    assert bg._dspark_prepare(SimpleNamespace(model=host), state, "rows", mx.array([1, 2])) is None
    assert len(seen) == 1
    host.mtp_box_prepare = lambda *a: None  # today's path
    state.copy_index = None
    state.controller = SimpleNamespace(cur=4, cost_policy=True, max_depth=4)
    plan = bg._dspark_prepare(SimpleNamespace(model=host), state, "rows", mx.array([1, 2], dtype=mx.uint32))
    assert plan[1] == 4 and plan[3] is True


def test_chain_resolves_box_drafts_and_stashes_the_verify(monkeypatch):
    """_run_verify_cycle_chain: a box-pending block is resolved before its verify; the verify's argmax is stashed."""
    cache = SimpleNamespace(offset=5000)
    resolved = []

    def resolve(gen_batch, state):
        resolved.append(True)
        state.box_pending = None
        state.drafts = mx.array([1, 2, 3], dtype=mx.uint32)
        state.draft_lps = [None] * 3

    model = SimpleNamespace(_omlx_dspark_decode_enabled=True, mtp_box_resolve=resolve)
    state = bg._MtpState(uid=1, chain=True, depth=4, mtp_cache=[], next_main=mx.array([9], dtype=mx.uint32),
                         drafts=None)
    state.box_pending = True

    def backbone(_model, inputs, _cache, **_):
        width = int(inputs.shape[1])
        assert inputs.tolist() == [[9, 1, 2, 3]]
        cache.offset += width
        rows = []
        for target in [1, 2, 7, 8]:
            row = [-100.0] * 16
            row[target] = 0.0
            rows.append(row)
        return mx.array([rows], dtype=mx.float32), mx.zeros((1, width, 8)), None

    drafted = []
    monkeypatch.setattr(bg, "_call_backbone", backbone)
    monkeypatch.setattr(bg, "_chain_rollback", lambda *a: True)
    monkeypatch.setattr(bg, "_chain_next_drafts", lambda gb, st, *a: drafted.append(st.last_verify[:3] + (
        st.last_verify[3].tolist(),)))
    monkeypatch.setattr(bg, "_clear_rollback", lambda _cache: None)
    batch = SimpleNamespace(model=model, prompt_cache=[cache], tokens=[list(range(5000))], samplers=[None],
                            fallback_sampler=lambda lp: mx.argmax(lp, axis=-1).astype(mx.uint32),
                            logits_processors=[], _token_context=[], max_tokens=[9000], _num_tokens=[0],
                            _matchers=[SimpleNamespace(advance=lambda token: False)])
    bg._run_verify_cycle_chain(batch, state)
    assert resolved == [True]
    assert drafted == [(2, [1, 2, 7, 8], [1, 2, 3], [9])]  # m = 2 (3 != 7); all 4 argmax; the verified row 0


def test_advance_groups_resolves_each_pair_before_its_inputs(monkeypatch):
    order = []

    class Host:
        _omlx_dspark_decode_enabled = True
        mtp_draft_jobs_enabled = False

        def mtp_verify_groups(self, lengths):
            order.append(("groups", lengths))
            return [[0, 1]]

        def mtp_box_resolve(self, row, state):
            order.append(("resolve", state.uid))
            state.box_pending, state.drafts = None, mx.array([state.uid], dtype=mx.uint32)

        def mtp_verify_requests(self, inputs, caches):
            order.append(("verify", [x.tolist() for x in inputs]))
            return [(mx.zeros((1, 2, 4)), mx.zeros((1, 2, 4)), None) for _ in inputs]

    states = {}
    for uid in (1, 2):
        st = SimpleNamespace(uid=uid, queue=deque(), chain=True, next_main=mx.array([9], dtype=mx.uint32),
                             drafts=None, box_pending=True, controller=None)
        states[uid] = st
    host = Host()
    batch = SimpleNamespace(uids=[1, 2], model=host, _token_context=[None, None])
    monkeypatch.setattr(bg, "_make_row_batch", lambda b, i, state=None: SimpleNamespace(
        model=host, prompt_cache=[i], _token_context=[None]))
    monkeypatch.setattr(bg, "_set_singleton_mrope_delta", lambda row: None)
    monkeypatch.setattr(bg, "_run_verify_cycle_chain", lambda row, state, **kw: order.append(("chain", state.uid)))
    monkeypatch.setattr(bg, "_replace_cache_rows", lambda b, r: None)
    monkeypatch.setattr(fused_batch.batched_head, "flush", lambda s: None)
    assert fused_batch._advance_groups(batch, SimpleNamespace(states=states), host)
    assert order == [("groups", [5, 5]), ("resolve", 1), ("resolve", 2), ("verify", [[[9, 1]], [[9, 2]]]),
                     ("chain", 1), ("chain", 2)]


def test_advance_dspark_declines_box_pending(monkeypatch):
    st = SimpleNamespace(queue=deque(), drafts=None, box_pending=True)
    batch = SimpleNamespace(uids=[1, 2])
    assert fused_batch._advance_dspark(batch, SimpleNamespace(states={1: st, 2: st}), None) is False


def test_reply_flags_drive_pf_disabled_and_ring(stats):
    ctl = stepd.BoxDraft()
    ctl.primed = True
    info = dict(flags=dw.R_PF_ACTIVE | dw.R_RING_OK, box_s=0.01, ndraft=4, drafter_ms=1.5)
    ctl.on_reply(info)
    assert ctl.pf_hold and ctl.primed
    for _ in range(stepd.PF_CLEAR):
        ctl.on_reply(dict(info, flags=dw.R_RING_OK))
    assert not ctl.pf_hold
    ctl.on_reply(dict(info, flags=dw.R_DISABLED))
    assert ctl.disabled and not ctl.primed and stats["box_draft_disabled_replies"] == 1
    ctl.on_reply(dict(info, flags=dw.R_RING_OK))
    assert not ctl.disabled
    for _ in range(stepd.RING_RETRIES):
        ctl.primed = True
        ctl.on_reply(dict(info, flags=0))
    assert ctl.sticky == "ring_not_ok"
    lost = stepd.BoxDraft()
    lost.on_lost(pipe_wire.StepdError({"error": "x", "code": "accept_mismatch"}))
    assert lost.sticky == "box_error"
    other = stepd.BoxDraft()
    other.on_lost(ConnectionError("link"))
    assert other.sticky is None
    other.on_reopen(SimpleNamespace(dspark=None))
    assert other.sticky == "lost_grant"


def _resolve_setup(monkeypatch, drafts):
    host = FakeHost()
    lm = og_model.OgLanguageModel.__new__(og_model.OgLanguageModel)
    for name in ("make_mtp_cache", "dspark_append_context"):
        object.__setattr__(lm, name, getattr(host, name))
    monkeypatch.setattr(og_model.OgLanguageModel, "_gpu_warm", lambda self: None)
    ctl = stepd.BoxDraft()
    ctl.ring.reset(np.zeros((3, 40, 512), np.uint16), 5040)
    ctl.ring.append(np.ones((3, stepd.TAP_DIM), np.uint16), 5040)
    ctl.mac_current = False
    encoder = FakeEncoder()
    encoder.stepd = ctl
    monkeypatch.setattr(ctl, "resolve", lambda enc, idle=None: drafts)
    monkeypatch.setattr(og_model, "session_of", lambda cache: encoder)
    state = SimpleNamespace(box_pending=True, next_main=mx.array([77], dtype=mx.uint32), mtp_cache=None,
                            hist_offset=5043, controller=SimpleNamespace(max_depth=4))
    gen_batch = SimpleNamespace(prompt_cache=[SimpleNamespace(size=lambda: 5043)])
    return lm, state, gen_batch, encoder, ctl


def test_resolve_uses_the_box_drafts(monkeypatch):
    lm, state, gen_batch, encoder, ctl = _resolve_setup(monkeypatch, [4, 5, 6])
    lm.mtp_box_resolve(gen_batch, state)
    assert state.box_pending is None and state.drafts.tolist() == [4, 5, 6] and state.draft_source == "box"
    assert state.draft_lps == [None] * 3 and state.mtp_cache is None and encoder.sent == []


def test_resolve_falls_back_to_the_mac_drafter(monkeypatch, stats):
    from omlx.patches.deepseek_v41 import dspark
    lm, state, gen_batch, encoder, ctl = _resolve_setup(monkeypatch, None)
    seen = {}

    def proposal(host, anchor, cache, width, cost_policy=False):
        seen.update(anchor=anchor.tolist(), offsets=[c.offset for c in cache], width=width, cost=cost_policy)
        return "logits", None

    def finish(gb, st, plan, logits, prev_buf):
        seen["plan"] = (plan[1], plan[3], logits, prev_buf)
        st.drafts = mx.array([8, 9], dtype=mx.uint32)
    monkeypatch.setattr(dspark, "proposal_forward", proposal)
    monkeypatch.setattr(bg, "_dspark_finish", finish)
    lm.mtp_box_resolve(gen_batch, state)
    assert seen == dict(anchor=[[77]], offsets=[5043] * 3, width=4, cost=True, plan=(4, True, "logits", None))
    assert ctl.mac_current and stats["box_draft_mac_rebuilds"] == 1
    kind, keep, anchor, kw = encoder.sent[-1]  # redraft: kickoff STEPD mode 1 at the same keep and anchor
    assert (kind, keep, anchor, kw["mode"], kw["explicit"]) == ("STEPD", 5043, 77, dw.MODE_EXPLICIT, [8, 9])


def test_fused_bit_follows_the_requests_own_verify(monkeypatch, stats):
    """V2.4: FUSED only when this request really verified in a fused pair (not merely >= FUSE_MIN sessions)."""
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000, sessions=3)
    monkeypatch.setattr(og_model.og_fused, "ENABLED", True)
    assert og_model._fused_regime()  # the old rule would have sent FUSED here
    og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd)
    assert encoder.sent[-1][3]["flags"] & dw.F_FUSED == 0
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000, sessions=3)
    encoder._og_fused = True
    og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd)
    assert encoder.sent[-1][3]["flags"] & dw.F_FUSED


def test_presend_one_row_block_is_a_step_with_taps(monkeypatch, stats):
    """V2.2: a Mac block STEPD mode 1 cannot carry goes out as a plain STEP carrying the committed rows' taps."""
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=950)
    ctl = encoder.stepd
    og_model._box_prepare(host, batch, state, hidden, committed, encoder, ctl)  # Mac era: stash for keep 953
    assert ctl.presend(encoder, 953, [99])
    assert encoder.sent[-1] == ("STEP", 953, [99], 3 * stepd.TAP_DIM * 2) and stats["box_draft_step_taps"] == 1
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=950)
    encoder.dspark = {k: v for k, v in dw.capability().items() if k != "step_taps"}  # a v1 grant
    og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd)
    assert not encoder.stepd.presend(encoder, 953, [99])  # og_model sends today's plain STEP


def test_filler_reply_is_verified_and_counted_as_a_fallback(stats):
    ctl = stepd.BoxDraft()
    ctl.box = dict(keep=5003, anchor=42, dmax=4, flags=0)
    held = []
    info = dict(mode_used=dw.MODE_BOX, flags=dw.R_DISABLED, L=2, ids=(42, 42), keep=5003, box_s=0.01, ndraft=4,
                drafter_ms=0.0)
    encoder = SimpleNamespace(recv_stepd=lambda idle=None: (info, b"rows", {}),
                              hold_ready=lambda *a: held.append(a))
    assert ctl.resolve(encoder) == [42] and held
    assert stats["box_draft_fillers"] == 1 and stats["box_draft_fallback_filler"] == 1 and stats["box_draft_cycles"] == 0
    ctl.box = dict(keep=5010, anchor=7, dmax=4, flags=0)
    info.update(flags=dw.R_DRAFTED | dw.R_RING_OK, L=4, ids=(7, 1, 2, 3), keep=5010)
    assert ctl.resolve(encoder) == [1, 2, 3] and stats["box_draft_cycles"] == 1 and stats["box_draft_drafted_tokens"] == 3


# ---- follow-up: extra draft sources per cycle, taps prefetch, send buffer (DSPARK-FOLLOWUP-MAC.md) -----------------
class FakeCopy:
    """copy_draft.PromptIndex surface _box_prepare uses: a token buffer, append, propose."""

    def __init__(self, tokens, proposal=None):
        self._buf, self.n, self.proposal, self.appended = np.array(tokens + [0] * 64, np.int64), len(tokens), proposal, []

    def append(self, ids):
        self.appended.append(list(ids))
        self._buf[self.n:self.n + len(ids)] = ids
        self.n += len(ids)

    def propose(self, budget):
        return self.proposal


class FakeTracker:
    def __init__(self, proposal=None):
        self.proposal, self.rows = proposal, []

    def append(self, ids, hidden):
        self.rows.append((list(ids), np.asarray(hidden[0, :, ::64].tolist(), dtype=np.float32)))

    def propose(self, tail, budget):
        return (list(self.proposal), "schema") if self.proposal else (None, None)


def test_extra_source_proposal_preempts_the_box_per_cycle(monkeypatch, stats):
    """A tool prompt's tracker proposes: STEPD mode 1 with its ids (taps attached), draft_source is its source, the
    tracker got this cycle's committed ids and exactly today's features, the copy index was appended once."""
    copy = FakeCopy(list(range(100, 5000)), proposal=[5, 6])
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000, copy=copy)
    tracker = state.extra_source_tracker = FakeTracker([7, 8, 9])
    assert og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd) is True
    kind, keep, anchor, kw = encoder.sent[-1]
    assert (kind, keep, kw["mode"], kw["explicit"], kw["nver"]) == ("STEPD", 5003, dw.MODE_EXPLICIT, [7, 8, 9], 4)
    assert kw["taps"] == np.array(hidden.reshape(-1, stepd.TAP_DIM).view(mx.uint16)).tobytes()
    assert state.drafts.tolist() == [7, 8, 9] and state.draft_source == "schema" and not getattr(state, "box_pending", None)
    (ids, feats), = tracker.rows
    assert ids == [21, 22, 99] and copy.appended == [[21, 22, 99]]
    today = np.asarray(hidden[0, :, ::64].tolist(), dtype=np.float32)  # what _dspark_prepare hands the tracker
    assert feats.tobytes() == today.tobytes()
    assert stats["box_draft_fallback_extra_source"] == 1 and encoder.stepd.counts["extra"] == 1
    assert encoder.stepd.sticky is None


def test_extra_source_without_proposal_falls_to_copy_then_box(monkeypatch, stats):
    copy = FakeCopy(list(range(100, 5000)), proposal=[5, 6])
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000, copy=copy)
    tracker = state.extra_source_tracker = FakeTracker(None)
    og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd)
    assert encoder.sent[-1][3]["explicit"] == [5, 6] and state.draft_source == "copy" and len(tracker.rows) == 1
    assert stats["box_draft_fallback_copy"] == 1 and not stats["box_draft_fallback_extra_source"]
    copy.proposal = None
    encoder._pending = None
    state.last_verify = (1, [5, 7, 8], [5, 6], mx.array([99], dtype=mx.uint32))  # verified [99, 5, 6], accept 1
    state.queue = deque([0, 0])
    committed2, hidden2 = mx.array([5, 7], dtype=mx.uint32), _taps(5003, 2)
    assert og_model._box_prepare(host, batch, state, hidden2, committed2, encoder, encoder.stepd) is True
    assert encoder.sent[-1][3]["mode"] == dw.MODE_BOX and state.box_pending and state.draft_source == "box"
    assert [ids for ids, _ in tracker.rows] == [[21, 22, 99], [5, 7]]  # appended on the box-drafted cycle too
    assert copy.appended == [[21, 22, 99], [5, 7]]


def test_extra_draft_mode_alone_is_not_sticky(monkeypatch, stats):
    """DS41_EXTRA_DRAFT=1 (production) with a prompt without tool schemas: no tracker, the box drafts."""
    from omlx.patches.deepseek_v41 import draft_sources
    monkeypatch.setattr(draft_sources, "MODE", "1")
    copy = FakeCopy(list(range(100, 5000)))
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000, copy=copy)
    host._ds41_draft_tokenizer = SimpleNamespace(decode=lambda ids: "plain chat, no schemas")
    assert og_model._box_capable(batch, state) is None
    assert og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd) is True
    assert state.extra_source_tracker is None and state.box_pending and encoder.stepd.sticky is None
    assert encoder.sent[-1][3]["mode"] == dw.MODE_BOX and not stats["box_draft_fallback_extra_source"]


def test_mac_era_leaves_tracker_and_copy_to_dspark_prepare(monkeypatch, stats):
    copy = FakeCopy(list(range(100, 950)), proposal=[5, 6])
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=950, copy=copy)
    tracker = state.extra_source_tracker = FakeTracker([7, 8, 9])
    assert og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd) is None
    assert tracker.rows == [] and copy.appended == []  # today's _dspark_prepare appends them next


def test_prefetch_taps_stash_is_used_for_the_stepd(monkeypatch, stats):
    """_remote's prefetch: taps evaluated with the verify; _box_prepare reads that stash (no second readback)."""
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000)
    verify_hidden = _taps(5000, 4)  # the verify's 4 rows; 3 commit
    og_model._prefetch_taps(encoder, 5000, 4, verify_hidden)
    start, bits = encoder._og_taps
    assert start == 5000 and bits.dtype == mx.uint16 and bits.shape == (1, 4, stepd.TAP_DIM)
    marker = (verify_hidden + 1).astype(mx.bfloat16)  # != the stash: proves which one the STEPD carried
    encoder._og_taps = (5000, marker.view(mx.uint16))
    og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd)
    assert encoder.sent[-1][3]["taps"] == np.array(marker[:, :3].reshape(-1, stepd.TAP_DIM).view(mx.uint16)).tobytes()
    assert encoder._og_taps is None
    # a stash for another verify start is ignored (today's readback of hidden_rows)
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000)
    encoder._og_taps = (4990, marker.view(mx.uint16))
    og_model._box_prepare(host, batch, state, hidden, committed, encoder, encoder.stepd)
    assert encoder.sent[-1][3]["taps"] == np.array(hidden.reshape(-1, stepd.TAP_DIM).view(mx.uint16)).tobytes()


def test_prefetch_taps_only_where_taps_are_read(stats):
    enc = SimpleNamespace(stepd=stepd.BoxDraft())
    og_model._prefetch_taps(enc, 500, 5, _taps(500, 5))
    assert enc._og_taps is None  # below 896 and nothing primed: today's path reads no taps
    og_model._prefetch_taps(enc, 892, 5, _taps(892, 5))
    assert enc._og_taps[0] == 892
    enc.stepd.stick("sampling")
    og_model._prefetch_taps(enc, 5000, 5, _taps(5000, 5))
    assert enc._og_taps is None
    plain = SimpleNamespace(stepd=None)
    og_model._prefetch_taps(plain, 5000, 5, _taps(5000, 5))
    assert plain._og_taps is None


def test_tap_features_equal_the_mlx_rows_bitwise():
    rng = np.random.default_rng(3)
    hidden = mx.array(rng.standard_normal((1, 5, stepd.TAP_DIM)).astype(np.float32) * 40).astype(mx.bfloat16)
    taps = np.array(hidden.reshape(-1, stepd.TAP_DIM).view(mx.uint16))
    got = og_model._tap_features(taps, None)
    assert got.shape == (1, 5, stepd.TAP_DIM) and got.dtype == np.float32
    want = np.asarray(hidden[0, :, ::64].tolist(), dtype=np.float32)
    assert np.asarray(got[0, :, ::64].tolist(), dtype=np.float32).tobytes() == want.tobytes()
    assert og_model._tap_features(None, hidden) is hidden


def test_send_buffer_only_on_a_grant(monkeypatch):
    calls = []
    sock = SimpleNamespace(setsockopt=lambda *a: calls.append(a))
    enc = pipe_wire.EncoderSession.__new__(pipe_wire.EncoderSession)
    enc.sock, enc.dspark_request = sock, None
    enc._grant({"dspark": dw.capability()})
    assert calls == []  # never asked: today's socket
    enc.dspark_request = {"ver": 1}
    enc._grant({"dspark_off": "not_loaded"})
    assert calls == [] and enc.dspark_off == "not_loaded"
    enc._grant({"dspark": dw.capability()})
    import socket
    assert calls == [(socket.SOL_SOCKET, socket.SO_SNDBUF, pipe_wire.SNDBUF)] and pipe_wire.SNDBUF == 1 << 20


def test_box_era_timings_and_send_before_host_ring(monkeypatch, stats):
    """prep/send timings are recorded, and the host ring gets the taps after the STEPD went out."""
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000)
    ctl = encoder.stepd
    order = []
    real_send, real_append = encoder.send_stepd, ctl.ring.append
    encoder.send_stepd = lambda *a, **k: (order.append("send"), real_send(*a, **k))[1]
    ctl.ring.append = lambda *a: (order.append("append"), real_append(*a))[1]
    og_model._box_prepare(host, batch, state, hidden, committed, encoder, ctl)
    assert order == ["send", "append"] and ctl.ring.offset == 5003
    assert stats["box_draft_prep_ms"] >= stats["box_draft_send_ms"] >= 0 and ctl.times["sent"] == 1
    assert "prep_ms" in ctl.summary() and "rtt_ms" not in ctl.summary()


def test_pf_active_mac_era_sends_no_taps_then_a_keys_only_ring(monkeypatch, stats):
    """PF_ACTIVE at c2: the Mac-drafted STEPD carries no taps (the box job is a plain step); when box drafting resumes
    the box ring is re-primed from the Mac ring's keys (K=128, T=0), then STEPD mode 0 appends as usual."""
    monkeypatch.setattr(stepd, "PF_TAPS", False)
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000, sessions=2)
    ctl = encoder.stepd
    og_model._box_prepare(host, batch, state, hidden, committed, encoder, ctl)  # box era: RING + STEPD mode 0
    encoder._pending, ctl.pf_hold = None, True
    committed2, hidden2 = mx.array([5, 7], dtype=mx.uint32), _taps(5003, 2)
    state.last_verify = (1, [5, 7, 8], [5, 6], mx.array([99], dtype=mx.uint32))  # verified [99, 5, 6], accept 1
    assert og_model._box_prepare(host, batch, state, hidden2, committed2, encoder, ctl) is None  # Mac era
    assert not ctl.primed and ctl.mac_current and ctl.ring.offset == 5005
    host.dspark_append_context(hidden2, state.mtp_cache)  # today's drafter: the Mac ring gets [5003, 5005)
    state.hist_offset = 5005
    assert ctl.presend(encoder, 5005, [7, 31, 32])
    kind, keep, anchor, kw = encoder.sent[-1]
    assert (kind, keep, kw["mode"], kw["nver"], kw["taps"]) == ("STEPD", 5005, dw.MODE_EXPLICIT, 3, b"")
    assert kw["flags"] & dw.F_NO_TAPS
    encoder._pending, ctl.pf_hold = None, False  # two clear replies later
    committed3, hidden3 = mx.array([31, 32, 9], dtype=mx.uint32), _taps(5005, 3)
    state.last_verify = (2, [31, 32, 9], [31, 32], mx.array([7], dtype=mx.uint32))  # verified [7, 31, 32], accept 2
    rings = sum(1 for s in encoder.sent if s[0] == "RING")
    assert og_model._box_prepare(host, batch, state, hidden3, committed3, encoder, ctl) is True
    ring, step = encoder.sent[-2:]
    assert ring[:4] == ("RING", 5005, 128, 0) and rings == 1  # 393 KB of Mac keys, not 128 tap rows (3.9 MB)
    assert step[3]["mode"] == dw.MODE_BOX and step[3]["taps"] == np.array(
        hidden3.reshape(-1, stepd.TAP_DIM).view(mx.uint16)).tobytes()


def test_pf_active_with_taps_keeps_the_ring(monkeypatch, stats):
    monkeypatch.setattr(stepd, "PF_TAPS", True)
    host, state, batch, encoder, committed, hidden = _setup(monkeypatch, base=5000, sessions=2)
    ctl = encoder.stepd
    og_model._box_prepare(host, batch, state, hidden, committed, encoder, ctl)
    encoder._pending, ctl.pf_hold = None, True
    state.last_verify = (1, [5, 7, 8], [5, 6], mx.array([99], dtype=mx.uint32))
    og_model._box_prepare(host, batch, state, _taps(5003, 2), mx.array([5, 7], dtype=mx.uint32), encoder, ctl)
    assert ctl.primed and ctl.presend(encoder, 5005, [7, 31, 32])
    kw = encoder.sent[-1][3]
    assert not kw["flags"] & dw.F_NO_TAPS and len(kw["taps"]) == 2 * stepd.TAP_DIM * 2
