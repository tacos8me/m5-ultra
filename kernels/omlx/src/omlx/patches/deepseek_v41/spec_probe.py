# SPDX-License-Identifier: MIT
"""Stats-only probe for copy-lock pre-send (PRESEND-FEASIBILITY.md, variant b). Sends nothing.

At every c1 cycle (exactly one box session) whose drafts came from the copy index, predict the STEP
that a copy-lock pre-send would send next, assuming all copy drafts are accepted and the source
continues (CopyIndex.speculate). When the real next STEP is sent (og_model.presend), compare
(keep, ids) byte for byte. A hit is a STEP whose box round trip a pre-send would have hidden.

Host work per cycle: a tuple compare before the send and one CopyIndex.speculate (one 5-token append
+ propose on a <=17 x 64 window) after it, i.e. while the box computes. No MLX sync, no box traffic.
DS41_OG_SPEC_PROBE=0 disables every call.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)
ENABLED = os.environ.get('DS41_OG_SPEC_PROBE', '1') == '1'
MAX_ROWS = 5
REASONS = ('partial', 'ended', 'dspark', 'copy_diff', 'concurrency', 'other')
COUNTS = ('c1_cycles', 'copy_cycles', 'would', 'hits', 'no_pred', 'unresolved', 'errors') + tuple(
    'miss_' + r for r in REASONS)
TIMES = ('c1_cycle_ms', 'c1_cycles_timed', 'hit_cycle_ms', 'hit_cycles_timed', 'hit_wait_ms', 'hit_waits')
STATS = {}


def bind(stats):
    """Use og_model.STATS (served at /og/stats) with spec_probe_* keys."""
    global STATS
    STATS = stats
    for key in COUNTS + TIMES:
        stats.setdefault('spec_probe_' + key, 0)
    stats['spec_probe_enabled'] = int(ENABLED)


class Record:
    """Per-request (per box session) probe state and counters."""

    __slots__ = ('pred', 'last', 'last_t', 'last_c1', 'last_hit', 'hit_keep', 'counts')

    def __init__(self):
        self.pred = None      # (keep, ids) a copy-lock pre-send would have sent next
        self.last = None      # (keep, ids) of the newest real STEP seen
        self.last_t = None
        self.last_c1 = False
        self.last_hit = False
        self.hit_keep = None  # keep of a hit STEP whose reply is not received yet
        self.counts = dict.fromkeys(COUNTS + TIMES, 0)


def _add(rec, key, value=1):
    rec.counts[key] += value
    STATS['spec_probe_' + key] = STATS.get('spec_probe_' + key, 0) + value


def record(encoder):
    rec = encoder.__dict__.get('_spec_probe')
    if rec is None:
        rec = encoder.__dict__['_spec_probe'] = Record()
    return rec


def classify(pred, keep, ids, source):
    """Why the real next STEP (keep, ids) differs from the predicted one."""
    pkeep, pids = pred
    if keep < pkeep:
        return 'partial'       # not every copy draft was accepted
    if keep != pkeep or not ids:
        return 'other'
    if ids[0] != pids[0]:
        return 'ended'         # all accepted, but the bonus left the copied span
    if source is None:
        return 'other'         # seen outside presend(): draft source unknown
    if source != 'copy':
        return 'dspark'        # same bonus, but the next block came from DSpark (or another source)
    return 'copy_diff'         # copy again, but a different source / length


def observe_step(encoder, ids, keep, source, c1):
    """The real STEP (keep, ids) about to be sent: resolve the pending prediction. Before the send; O(rows)."""
    rec = record(encoder)
    ids = tuple(ids)
    if rec.last == (keep, ids):
        return
    now = time.perf_counter()
    if rec.last_t is not None and rec.last_c1 and c1:
        ms = (now - rec.last_t) * 1000.0
        _add(rec, 'c1_cycle_ms', ms)
        _add(rec, 'c1_cycles_timed')
        if rec.last_hit:
            _add(rec, 'hit_cycle_ms', ms)
            _add(rec, 'hit_cycles_timed')
    rec.last_hit = False
    if c1:
        _add(rec, 'c1_cycles')
    pred, rec.pred = rec.pred, None
    if pred is not None:
        if not c1:
            _add(rec, 'miss_concurrency')
        elif pred == (keep, ids):
            _add(rec, 'hits')
            rec.last_hit = True
            rec.hit_keep = keep
        else:
            _add(rec, 'miss_' + classify(pred, keep, ids, source))
    rec.last, rec.last_t, rec.last_c1 = (keep, ids), now, c1


def predict(encoder, state, budget, keep, ids, c1):
    """After the real STEP was sent: the STEP a copy-lock pre-send would send next (or none)."""
    rec = record(encoder)
    if not c1 or getattr(state, 'draft_source', None) != 'copy':
        return
    _add(rec, 'copy_cycles')
    copy = getattr(state, 'copy_index', None)
    pending = getattr(copy, 'pending', None)
    if pending is None or pending[1] != len(ids) - 1:
        _add(rec, 'errors')
        return
    spec = copy.speculate(budget)
    if spec is None or 1 + len(spec[1]) > MAX_ROWS:
        _add(rec, 'no_pred')
        return
    bonus, drafts = spec
    rec.pred = (keep + len(ids), (int(bonus),) + tuple(int(x) for x in drafts))
    _add(rec, 'would')


def note_recv(encoder, keep, timing):
    """A STEP reply arrived: the box wait of a hit STEP is what a pre-send would have hidden."""
    rec = encoder.__dict__.get('_spec_probe')
    if rec is None or rec.hit_keep is None or rec.hit_keep != keep:
        return
    rec.hit_keep = None
    _add(rec, 'hit_wait_ms', 1000.0 * float(timing.get('wait_s') or 0.0))
    _add(rec, 'hit_waits')


def summary(counts):
    c = counts
    def mean(total, n):
        return round(total / n, 2) if n else None
    return dict(
        c1_cycles=c['c1_cycles'], copy_cycles=c['copy_cycles'], would=c['would'], hits=c['hits'],
        hit_pct=round(100.0 * c['hits'] / c['c1_cycles'], 1) if c['c1_cycles'] else None,
        miss={r: c['miss_' + r] for r in REASONS}, no_pred=c['no_pred'], unresolved=c['unresolved'],
        errors=c['errors'], c1_cycle_ms=mean(c['c1_cycle_ms'], c['c1_cycles_timed']),
        hit_cycle_ms=mean(c['hit_cycle_ms'], c['hit_cycles_timed']),
        hit_wait_ms=mean(c['hit_wait_ms'], c['hit_waits']))


def close(encoder, request_id):
    """The request ended: one summary line in og-child.log."""
    rec = encoder.__dict__.pop('_spec_probe', None)
    if rec is None:
        return
    if rec.pred is not None:
        _add(rec, 'unresolved')
    s = summary(rec.counts)
    logger.info('ds41-og spec-probe %s: c1_cycles=%d copy_cycles=%d would=%d hits=%d hit_pct=%s '
                'miss[partial=%d,ended=%d,dspark=%d,copy_diff=%d,concurrency=%d,other=%d] no_pred=%d unresolved=%d '
                'errors=%d cycle_ms[c1=%s,hit=%s] hit_wait_ms=%s',
                request_id, s['c1_cycles'], s['copy_cycles'], s['would'], s['hits'], s['hit_pct'],
                *[s['miss'][r] for r in REASONS], s['no_pred'], s['unresolved'], s['errors'],
                s['c1_cycle_ms'], s['hit_cycle_ms'], s['hit_wait_ms'])
