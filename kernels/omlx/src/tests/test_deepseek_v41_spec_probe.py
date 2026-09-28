# SPDX-License-Identifier: MIT
"""ds41-probe: copy-lock pre-send prediction (CopyIndex.speculate) and the stats-only probe."""

import logging
import random
from types import SimpleNamespace

import numpy as np
import pytest

from omlx.patches.mlx_lm_mtp import copy_draft


def _full_state(index):
    return (
        index.n, index._buf.tobytes(), len(index._buf), index._order.tobytes(), index._sorted.tobytes(),
        [(k, list(v)) for k, v in index._extra.items()],
        index.cur, index.min_match, index.expected, index.pending,
    )


def _stream(rng, vocab=40, prompt_len=600, out_len=900):
    """A prompt and a target continuation that copies spans of it (and of itself) between noise."""
    prompt = [rng.randrange(vocab) for _ in range(prompt_len)]
    out = []
    while len(out) < out_len:
        if rng.random() < 0.6:
            src = prompt + out
            a = rng.randrange(max(1, len(src) - 60))
            out += src[a:a + rng.randint(5, 60)]
        else:
            out += [rng.randrange(vocab) for _ in range(rng.randint(1, 8))]
    return prompt, out[:out_len]


def _drive(seed, speculate):
    """The served acceptance loop against a fixed target stream; returns proposals + probe outcomes."""
    rng = random.Random(seed)
    prompt, target = _stream(rng)
    full = prompt + target
    pos = len(prompt) - 1  # the anchor (last committed token) is full[pos]
    index = copy_draft.CopyIndex(full[:pos + 1])
    budget = len(target) + 10
    proposals, outcomes = [], []
    pred = None
    while pos + 6 < len(full) and budget > 6:
        drafts = index.propose(budget)
        proposals.append(drafts)
        if pred is not None:  # the previous cycle's prediction, checked against this real proposal
            outcomes.append((pred, drafts))
            pred = None
        if drafts is None:  # DSpark cycle: some accepted tokens + a correction/bonus, no copy observe
            step = rng.randint(1, 5)
            index.append(full[pos + 1:pos + 1 + step])
            pos += step
            budget -= step
            continue
        source, k = index.pending
        if speculate:
            spec = index.speculate(budget)
            virtual = full[:pos + 1] + list(index._buf[source:source + k])
            g = virtual[source + k]
            pred = (spec, g, k)
        m = 0
        while m < k and drafts[m] == full[pos + 1 + m]:
            m += 1
        bonus = full[pos + 1 + m]
        index.observe(m)
        index.append(full[pos + 1:pos + m + 2])
        pos += m + 1
        budget -= m + 1
        if speculate:
            spec, g, k0 = pred
            pred = (spec, g, k0, m, bonus)
    return proposals, outcomes, _full_state(index)


@pytest.mark.parametrize("seed", range(12))
def test_speculate_predicts_the_next_copy_block_exactly(seed):
    _, outcomes, _ = _drive(seed, speculate=True)
    held = hits = 0
    for (spec, g, k, m, bonus), real in outcomes:
        if m == k and bonus == g:  # the hypothesis held: the prediction must be the real proposal
            held += 1
            assert (None if spec is None else spec[1]) == real
            if spec is not None:
                assert spec[0] == bonus
                hits += 1
        elif spec is not None and real is not None and m == k:
            assert spec[0] != bonus
    assert held > 5 and hits > 5


@pytest.mark.parametrize("seed", range(12))
def test_speculate_changes_nothing(seed):
    a = _drive(seed, speculate=False)
    b = _drive(seed, speculate=True)
    assert a[0] == b[0]  # every served proposal
    assert a[2] == b[2]  # the full final index state (buffer, n-gram tables, policy)


def test_speculate_restores_every_field():
    rng = random.Random(7)
    prompt, target = _stream(rng)
    index = copy_draft.CopyIndex(prompt)
    checked = 0
    for i in range(0, 400, 3):
        index.append(target[i:i + 3])
        if index.propose(8) is None:
            continue
        before = _full_state(index)
        index.speculate(8)
        index.speculate(2)
        index.speculate(0)
        assert _full_state(index) == before
        checked += 1
    assert checked > 20


def test_speculate_periodic_self_overlap():
    # A period-3 loop: the copy source is the latest occurrence, so the copied span ends at n and the
    # bonus comes from the drafts themselves (source + k == n).
    index = copy_draft.CopyIndex([5, 6, 7] * 10)
    drafts = index.propose(20)
    source, k = index.pending
    assert source + k == index.n
    bonus, nxt = index.speculate(20)
    assert bonus == drafts[0]
    index.observe(k)
    index.append(drafts + [bonus])
    assert index.propose(20 - k - 1) == nxt


def test_speculate_without_pending_or_budget():
    index = copy_draft.CopyIndex([1, 2, 3, 4, 5, 6, 7, 8])
    assert index.speculate(10) is None
    index = copy_draft.CopyIndex([1, 2, 3, 4, 9, 1, 2, 3, 4])
    assert index.propose(10) is not None
    assert index.speculate(5) is None  # budget after the 5 hypothetical tokens is 0


# ---- spec_probe bookkeeping ------------------------------------------------------------------------

@pytest.fixture
def probe(monkeypatch):
    pytest.importorskip("mlx.core")
    from omlx.patches.deepseek_v41 import spec_probe
    stats = {}
    spec_probe.bind(stats)
    return spec_probe, stats


class _Copy:
    def __init__(self, spec, k=4):
        self.pending, self.spec = (0, k), spec

    def speculate(self, budget):
        return self.spec


def test_probe_counts_hits_and_miss_reasons(probe, caplog):
    sp, stats = probe
    enc = SimpleNamespace()
    def predict(spec, budget, keep, ids, c1=True, source='copy'):
        state = SimpleNamespace(draft_source=source, copy_index=_Copy(spec, k=len(ids) - 1))
        sp.predict(enc, state, budget, keep, ids, c1)
    # cycle 1: copy block at keep 100; predicts keep 105 [9, 1, 2, 3, 4]
    sp.observe_step(enc, [8, 5, 6, 7, 8], 100, 'copy', True)
    predict((9, [1, 2, 3, 4]), 50, 100, [8, 5, 6, 7, 8])
    # cycle 2: the exact STEP -> hit; predicts 110 [5, ...]
    sp.observe_step(enc, [9, 1, 2, 3, 4], 105, 'copy', True)
    sp.observe_step(enc, [9, 1, 2, 3, 4], 105, None, True)  # _remote sees the same STEP: no double count
    sp.note_recv(enc, 105, dict(wait_s=0.008))
    predict((5, [6, 7, 8, 9]), 45, 105, [9, 1, 2, 3, 4])
    # cycle 3: only 2 accepted -> partial
    sp.observe_step(enc, [1, 2, 3], 108, 'dspark', True)
    predict(None, 40, 108, [1, 2, 3], source='dspark')
    # cycle 4 (dspark, no prediction), then copy with a prediction that ends
    sp.observe_step(enc, [4, 5, 6, 7, 8], 111, 'copy', True)
    predict((1, [2, 3, 4, 5]), 35, 111, [4, 5, 6, 7, 8])
    sp.observe_step(enc, [7, 3, 3, 3, 3], 116, 'copy', True)  # bonus 7 != 1 -> ended
    predict((1, [2, 3, 4, 5]), 30, 116, [7, 3, 3, 3, 3])
    sp.observe_step(enc, [1, 9, 9], 121, 'dspark', True)  # same bonus, DSpark block -> dspark
    predict(None, 25, 121, [1, 9, 9], source='dspark')  # not copy: nothing
    sp.observe_step(enc, [3, 3, 3, 3, 3], 124, 'copy', True)
    predict(None, 20, 124, [3, 3, 3, 3, 3])  # copy, but no copy block next
    sp.observe_step(enc, [3, 4], 129, 'dspark', True)
    predict((3, [1]), 15, 129, [3, 4], c1=False)  # not c1: no prediction
    c = enc._spec_probe.counts
    assert (c['c1_cycles'], c['copy_cycles'], c['would'], c['hits'], c['no_pred']) == (8, 5, 4, 1, 1)
    assert (c['miss_partial'], c['miss_ended'], c['miss_dspark']) == (1, 1, 1)
    assert c['hit_waits'] == 1 and abs(c['hit_wait_ms'] - 8.0) < 1e-9
    assert c['hit_cycles_timed'] == 1 and c['c1_cycles_timed'] == 7 and c['errors'] == 0
    assert stats['spec_probe_hits'] == 1 and stats['spec_probe_would'] == 4
    with caplog.at_level(logging.INFO, logger=sp.__name__):
        sp.close(enc, 'req-1')
    line = [r.getMessage() for r in caplog.records if 'spec-probe req-1' in r.getMessage()]
    assert line and 'hits=1' in line[0] and 'miss[partial=1,ended=1,dspark=1' in line[0]
    assert '_spec_probe' not in enc.__dict__


def test_probe_concurrency_and_unresolved(probe):
    sp, stats = probe
    enc = SimpleNamespace()
    st = SimpleNamespace(draft_source='copy', copy_index=_Copy((9, [1, 2, 3, 4])))
    sp.observe_step(enc, [8, 5, 6, 7, 8], 100, 'copy', True)
    sp.predict(enc, st, 50, 100, [8, 5, 6, 7, 8], True)
    sp.observe_step(enc, [9, 1, 2, 3, 4], 105, 'copy', False)  # a second session opened meanwhile
    assert enc._spec_probe.counts['miss_concurrency'] == 1 and enc._spec_probe.counts['hits'] == 0
    sp.observe_step(enc, [9, 1], 110, 'copy', True)
    sp.predict(enc, st, 45, 110, [9, 1, 2, 3, 4][:2], True)  # pending k=4 != 1 draft: error, no prediction
    assert enc._spec_probe.counts['errors'] == 1
    st2 = SimpleNamespace(draft_source='copy', copy_index=_Copy((9, [1, 2, 3, 4]), k=1))
    sp.predict(enc, st2, 45, 110, [9, 1], True)
    sp.close(enc, 'req-2')
    assert stats['spec_probe_unresolved'] == 1
