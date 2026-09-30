# SPDX-License-Identifier: MIT
"""DSpark drafting on the RTX box (STEPD-SPEC.md v1, FROZEN): the Mac verifies, the box drafts.

DS41_OG_BOX_DRAFT=1 turns it on (default 0: OPEN never asks for the grant, nothing here runs, and the wire bytes are
today's). With the ACK grant, a request that is greedy, has no logit processors and no extra draft source is
"box-capable". From 896 committed tokens on, the Mac keeps the box ring current: RING once (the Mac's DSpark keys),
then every step carries the committed rows' BF16 taps (STEPD). From 1024 tokens (the Mac cost policy's own threshold)
the box drafts: after each verify the Mac sends STEPD mode 0 (verify argmax + taps) instead of running its drafter, and
the reply (STPD) carries the drafts and the layer 0-19 rows of [anchor] + drafts for the next verify.

Outputs: the Mac still verifies every token, and drafts only change how many rows a cycle verifies. Box-drafted
widths are 2..5 rows (depth 1..4), the range the Mac verify is M-invariant on at >= 1024, so T=0 text is the same
with the flag on or off. A reply the Mac must not verify (a bonus-only STPD: an L=1 verify is bitwise different from
the served L >= 2 path) is dropped, and the Mac drafts that block itself (redraft).

The Mac drafts instead (reason in /og/stats box_draft_fallback_*): below 1024 tokens, T > 0, logit processors,
PF_ACTIVE with >= 2 sessions (§13), the kill switch (DISABLED), bonus-only replies, hard box errors and a reopen without
the grant (the rest of that request then stays on today's path). Per cycle, a Mac proposal pre-empts the box draft
exactly as it pre-empts the Mac drafter today: the tool-schema / retrieval extra source (draft_sources.py, tool prompts
only) first, then the copy index; the proposal goes out as STEPD mode 1 (box_draft_fallback_extra_source / _copy).
No MLX import here: og_serve/test_stepd.py drives this module against a CPU fake box.
"""

import logging
import os
import struct
import time

import numpy as np

try:
    from . import dspark_wire as dw
    from . import pipe_wire
except ImportError:  # loaded standalone by file path (og_serve CPU tests)
    import importlib.util as _util
    import sys as _sys
    from pathlib import Path as _Path

    def _load(name):
        if name in _sys.modules:
            return _sys.modules[name]
        spec = _util.spec_from_file_location(name, _Path(__file__).with_name(name + '.py'))
        module = _util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _sys.modules[name] = module
        return module

    dw = _load('dspark_wire')
    pipe_wire = _load('pipe_wire')

logger = logging.getLogger(__name__)

ENABLED = os.environ.get('DS41_OG_BOX_DRAFT', '0') == '1'
PRIME_CTX = 896  # §10.3: from here the box ring is kept current (RING, then taps with every STEPD)
MIN_CTX = dw.MIN_CTX  # mode 0 only when the committed context before the cycle is >= 1024 (the Mac's cost policy)
RING = dw.RING
TAP_DIM = dw.TAP_DIM
KEY_DIM = dw.KEY_DIM
STAGES = dw.STAGES
WIDTH = dw.WIDTH
PF_CLEAR = 2  # §13: back to box drafting after this many consecutive replies without PF_ACTIVE
# PF_ACTIVE at >= 2 sessions: the Mac drafts, and its STEPD mode 1 carries no taps (NO_TAPS), so during another
# session's prefill the box job is exactly a plain step (no ~100 KB upload, no ring append), i.e. today's decode; box
# drafting resumes after a keys-only RING from the Mac's own ring (393 KB, once per prefill). 1 = attach taps (the
# ring stays current and needs no re-prime, but every step uploads and appends while the prefill runs).
PF_TAPS = os.environ.get('DS41_OG_BOX_PF_TAPS', '0') == '1'
RING_RETRIES = 3  # consecutive re-primes the box answers with ring_ok = 0 (kill switch off) before giving up

REASONS = ('no_grant', 'sampling', 'processors', 'extra_source', 'cost_policy', 'short_ctx', 'copy', 'pf_active',
           'disabled', 'ring_not_ok', 'filler', 'bonus_only', 'box_error', 'lost_grant', 'clamp', 'ring_gap')
COUNTS = ('asked', 'granted', 'requests', 'steps', 'mode0', 'mode1', 'kickoffs', 'cycles', 'drafted_tokens',
          'fillers', 'step_taps', 'redrafts', 'rings', 'ring_resends', 'ring_rows', 'ring_bytes', 'tap_rows',
          'tap_bytes', 'mac_rebuilds', 'replies', 'pf_active_replies', 'disabled_replies', 'recoveries', 'drained',
          'errors', 'send_errors')
# prep_ms: hook entry -> STEPD handed to the kernel (the Mac's own critical path after the verify's sync); send_ms:
# encode + sendall; rtt_ms: STEPD send -> STPD received (box_ms of it is the box job); wait_ms: the Mac blocked on it.
TIMES = ('wait_ms', 'drafter_ms', 'box_ms', 'prep_ms', 'send_ms', 'rtt_ms')
STATS = {}


def bind(stats):
    """Use og_model.STATS (/og/stats) with box_draft_* keys."""
    global STATS
    STATS = stats
    for key in COUNTS + TIMES + tuple('fallback_' + r for r in REASONS):
        stats.setdefault('box_draft_' + key, 0)
    stats['box_draft_enabled'] = int(ENABLED)


def count(key, value=1):
    STATS['box_draft_' + key] = STATS.get('box_draft_' + key, 0) + value


def open_request(pipe_costs, fused_costs):
    """OPEN.dspark (§3) from the Mac's PipelineDepthController tables ((floor, (c2, c3, c4, c5)), ...)."""
    request = {'ver': dw.VER, 'costs': {name: [[int(floor), [float(c) for c in costs]] for floor, costs in table]
                                        for name, table in (('pipe', pipe_costs), ('fused', fused_costs))}}
    dw.parse_costs(request['costs'])  # a malformed DS41_PIPE_COSTS* override fails here, not as a box bad_request
    return request


def wants(params, prompt_tokens):
    """Ask for the grant at OPEN: greedy, no penalties/budget/grammar, and able to reach MIN_CTX (§10.1).

    A heuristic on the request's sampling params: the per-cycle check (eligible()) is the authority, so a false
    positive only holds a box ring slot and a false negative only keeps the request on the Mac drafter."""
    if not ENABLED or params is None:
        return False
    return (float(getattr(params, 'temperature', 1.0) or 0.0) == 0.0
            and float(getattr(params, 'repetition_penalty', 1.0) or 1.0) == 1.0
            and not getattr(params, 'presence_penalty', 0.0) and not getattr(params, 'frequency_penalty', 0.0)
            and getattr(params, 'thinking_budget', None) is None
            and getattr(params, 'compiled_grammar', None) is None
            and prompt_tokens + int(getattr(params, 'max_tokens', 0) or 0) > MIN_CTX)


class HostRing:
    """The Mac's copy of the box ring (§10.4): DSpark keys up to keys_end (a snapshot of the Mac's GPU ring, BF16
    bits, chronological) and the BF16 taps of every committed position after it, the newest RING kept (~3.9 MB)."""

    def __init__(self):
        self.keys = np.zeros((STAGES, 0, KEY_DIM), np.uint16)
        self.keys_end = 0
        self.taps = np.zeros((RING, TAP_DIM), np.uint16)
        self.offset = 0  # position after the newest committed row held

    def reset(self, keys, keys_end):
        keys = np.asarray(keys, np.uint16).reshape(STAGES, -1, KEY_DIM)[:, -RING:]
        if keys.shape[1] > keys_end:
            raise ValueError(f'{keys.shape[1]} key rows end at position {keys_end}')
        self.keys, self.keys_end, self.offset = np.ascontiguousarray(keys), int(keys_end), int(keys_end)

    def append(self, taps, start):
        taps = np.asarray(taps, np.uint16).reshape(-1, TAP_DIM)
        if start != self.offset:
            raise ValueError(f'host ring gap: append at {start}, ring ends at {self.offset}')
        n = taps.shape[0]
        slots = np.arange(start, start + n) % RING
        self.taps[slots[-RING:]] = taps[-RING:]
        self.offset += n

    def window(self, end=None):
        """(keys [STAGES, K, KEY_DIM], taps [T, TAP_DIM]) for positions [end - K - T, end), K + T <= RING; end
        defaults to offset (every row held)."""
        end = self.offset if end is None else int(end)
        if not self.keys_end <= end <= self.offset:
            raise ValueError(f'host ring holds [{self.keys_end}, {self.offset}), not {end}')
        t = min(RING, end - self.keys_end)
        k = min(self.keys.shape[1], RING - t)
        if end - t < self.offset - RING:
            raise ValueError(f'host ring rows before {self.offset - RING} are overwritten')
        taps = self.taps[np.arange(end - t, end) % RING]
        return np.ascontiguousarray(self.keys[:, self.keys.shape[1] - k:]), taps


class BoxDraft:
    """One request's box-drafting state (EncoderSession.stepd); every method runs on the engine thread."""

    def __init__(self):
        self.ring = HostRing()
        self.primed = False  # the box ring holds ring[.., ring.offset): RING sent, no loss/gap reported since
        self.mac_current = True  # the Mac GPU DSpark ring holds every committed position (the Mac drafted last)
        self.sticky = None  # why the rest of this request stays on today's path (plain STEP, Mac drafter)
        self.pf_hold, self.pf_clear = False, 0
        self.disabled = False  # the newest reply carried DISABLED (kill switch or drafter not loaded)
        self.ring_fails = 0
        self.explicit = None  # Mac era: this cycle's STEPD mode 1 (sent by presend once the Mac has drafted)
        self.box = None  # the mode 0 STEPD in flight
        self.owner = None  # the MTP state this ring follows (a new state restarts from the Mac ring)
        self.counts = dict.fromkeys(('box', 'copy', 'extra', 'mac', 'filler', 'redraft', 'rings', 'recoveries'), 0)
        # box-era sums (ms) for the per-request log: prep/send over box-era STEPDs, rtt/box/wait over resolved replies
        self.times = dict.fromkeys(('prep', 'send', 'sent', 'rtt', 'box', 'wait', 'resolved'), 0.0)
        count('requests')

    # ---- replies and session loss (pipe_wire calls these) ------------------------------------------------------
    def on_reply(self, info):
        """Every STPD: fairness (§13), kill switch (§12) and ring status (§6.3)."""
        flags = info['flags']
        count('replies')
        count('box_ms', info['box_s'] * 1000.0)
        if info.get('ndraft'):
            count('drafter_ms', info['drafter_ms'])
        if flags & dw.R_PF_ACTIVE:
            self.pf_hold, self.pf_clear = True, 0
            count('pf_active_replies')
        elif self.pf_hold:
            self.pf_clear += 1
            if self.pf_clear >= PF_CLEAR:
                self.pf_hold = False
        self.disabled = bool(flags & dw.R_DISABLED)
        if self.disabled:
            self.primed = False  # the box ignores RING and taps while off: re-prime once a reply comes without it
            count('disabled_replies')
        elif flags & dw.R_RING_OK:
            self.ring_fails = 0
        elif self.primed:
            self.primed = False
            self.ring_fails += 1
            count('fallback_ring_not_ok')
            if self.ring_fails >= RING_RETRIES:
                self.stick('ring_not_ok')

    def on_lost(self, error):
        """pipe_wire.recover: the box session is gone (the new one has no ring). A hard ERR to a STEPD is a bug on
        one side (§11): the rest of this request goes back to today's path; the reopen restores its tokens."""
        self.primed, self.box = False, None
        self.counts['recoveries'] += 1
        count('recoveries')
        code = getattr(error, 'code', None)
        if code is not None:
            count('errors')
            logger.warning('ds41-og box drafting: box error %s (%s); this request continues on the Mac drafter',
                           code, error)
            self.stick('box_error')

    def on_reopen(self, encoder):
        self.primed = False
        if encoder.dspark is None and self.sticky is None:
            self.stick('lost_grant')

    def stick(self, reason):
        if self.sticky is None:
            self.sticky = reason
            count('fallback_' + reason)
        self.primed = False

    # ---- one cycle, after the verify's accept (og_model.mtp_box_prepare) --------------------------------------
    def prepare(self, encoder, *, owner, base, n, anchor, verify, taps_fn, mac_keys_fn, rebuild_fn, propose_fn,
                sessions, fused, budget, t0=None):
        """Positions [base, keep = base + n) were just committed; `anchor` sits at keep. Returns what this cycle does:

        ('box', None): STEPD mode 0 sent; resolve() returns the drafts before the next verify.
        ('copy', ids) / ('extra', ids): STEPD mode 1 sent with the copy index's / the extra source's ids (the next
        verify's drafts).
        ('mac', None): the Mac drafts as today (its GPU ring is current); presend() sends the block as STEPD mode 1.
        ('plain', None): today's path end to end (plain STEP).

        verify: (nver, a, argmax, ids_prev) of the verify that committed these rows, None for the first block after
        admission (a kickoff, nver = 0). taps_fn() -> [n, TAP_DIM] BF16 bits of the committed rows; mac_keys_fn() ->
        (keys [STAGES, K, KEY_DIM], end) of the Mac GPU ring; rebuild_fn(keys, taps, end) rebuilds that ring from the
        host ring; propose_fn() -> (ids, source) of this cycle's Mac proposal ((None, None): the box drafts), called
        once per box-era cycle only (the Mac era runs today's copy / extra-source code). t0: the hook's entry time
        (perf_counter), for prep_ms.
        """
        t0 = time.perf_counter() if t0 is None else t0
        if owner is not self.owner:
            self.owner, self.mac_current = owner, True
        keep = base + n
        if self.sticky is not None or encoder.dspark is None:
            return self._plain(rebuild_fn, base)
        if verify is not None:
            _, a, argmax, ids_prev = verify
            if a != n - 1 or dw.accept_count(ids_prev, argmax) != a or argmax[a] != anchor:
                count('fallback_clamp')  # a truncated commit ends the request: no STEPD for it (§6.1)
                return self._plain(rebuild_fn, base)
        if not self.ring.offset and keep < PRIME_CTX:
            return 'plain', None  # below 896: today's path, nothing collected
        taps = np.asarray(taps_fn(), np.uint16).reshape(n, TAP_DIM)
        box_era = base >= MIN_CTX and not self.disabled and not (self.pf_hold and sessions >= 2)
        if self.ring.offset != base:
            if not self.mac_current:
                self.stick('ring_gap')
                return 'plain', None
            keys, end = mac_keys_fn()
            if end != base:
                self.stick('ring_gap')
                return 'plain', None
            self.ring.reset(keys, end)
            self.primed = False
        elif box_era and not self.primed and self.mac_current:
            # A re-prime after the Mac drafted (PF_ACTIVE, the kill switch): its ring's keys (393 KB), not 128 taps rows
            keys, end = mac_keys_fn()
            if end == base:
                self.ring.reset(keys, end)
        count('tap_rows', n)
        if box_era:
            ids, source = propose_fn()  # a Mac proposal decides first, as it does over the Mac drafter (DSPARK-BOX §2.4)
            appended = self._ring_first(encoder, verify, taps, base)
            flags = dw.F_FUSED if fused else 0
            self.mac_current = False  # the Mac ring is not appended while the box drafts; rebuilt when needed
            if ids:
                kind = 'copy' if source in (None, 'copy') else 'extra'
                self._send(encoder, keep, anchor, verify, taps, dw.MODE_EXPLICIT, list(ids), 0, flags, t0)
                self.counts[kind] += 1
                count('fallback_copy' if kind == 'copy' else 'fallback_extra_source')
                action = kind, list(ids)
            else:
                dmax = max(1, min(WIDTH, int(budget)))  # >= 1: a box draft keeps the verify at L >= 2
                self._send(encoder, keep, anchor, verify, taps, dw.MODE_BOX, (), dmax, flags, t0)
                self.box = dict(keep=keep, anchor=anchor, dmax=dmax, flags=flags)
                self.counts['box'] += 1
                action = 'box', None
            if not appended:  # after the send: the host ring is not on the STEPD's critical path
                self.ring.append(taps, base)
            return action
        reason = 'short_ctx' if base < MIN_CTX else 'disabled' if self.disabled else 'pf_active'
        count('fallback_' + reason)
        self._mac_ring(rebuild_fn, base)
        no_taps = reason == 'pf_active' and not PF_TAPS
        if no_taps:  # the box ring gets a gap; re-primed from the Mac keys when box drafting resumes
            self.primed = False
            self.ring.append(taps, base)
        else:
            self._ring_then_append(encoder, verify, taps, base)
        self.explicit = dict(keep=keep, verify=verify, taps=taps, fused=fused, no_taps=no_taps)
        self.counts['mac'] += 1
        return 'mac', None

    def presend(self, encoder, keep, ids):
        """og_model.presend / _remote in the Mac era: the Mac's block as STEPD mode 1 with this cycle's taps, or, for
        a block mode 1 cannot carry, a plain STEP with them (V2.2). False: nothing prepared for (keep), or a v1 grant
        without step_taps (the caller sends today's plain STEP)."""
        stash, self.explicit = self.explicit, None
        if stash is None or stash['keep'] != keep or self.sticky is not None or encoder.dspark is None:
            return False
        if 2 <= len(ids) <= 1 + dw.MAX_EXPLICIT:
            no_taps = stash['no_taps'] and stash['verify'] is not None
            flags = (dw.F_FUSED if stash['fused'] else 0) | (dw.F_NO_TAPS if no_taps else 0)
            self._send(encoder, keep, ids[0], stash['verify'], stash['taps'], dw.MODE_EXPLICIT, list(ids[1:]), 0, flags)
            return True
        taps = b''
        if stash['verify'] is not None and not self.disabled and not stash['no_taps']:  # a kickoff's rows went with its RING
            if not encoder.dspark.get('step_taps'):
                return False
            taps = stash['taps'].tobytes()
        if not 1 <= len(ids) <= 5:
            return False
        try:
            encoder.send_step(list(ids), keep, taps=taps)
        except OSError:  # the step is recorded as in flight: its receive fails and rebuilds the session
            count('send_errors')
        if taps:
            count('step_taps')
        return True

    def ensure_mac_ring(self, rebuild_fn, end=None):
        """The Mac drafts a block the box did not (resolve() returned None): make its GPU ring current through
        `end` (default: every committed position the host ring holds)."""
        if not self.mac_current:
            end = self.ring.offset if end is None else end
            rebuild_fn(*self.ring.window(end), end)
            self.mac_current = True
            count('mac_rebuilds')

    def pending_explicit(self, keep):
        return self.explicit is not None and self.explicit['keep'] == keep

    # ---- the next verify needs the box's drafts (og_model.mtp_box_resolve) -----------------------------------
    def resolve(self, encoder, idle=None):
        """The drafts of the mode 0 STEPD in flight (1..4 ids; their rows wait in the encoder for the verify), or
        None: the Mac drafts this block itself and then calls redraft(). A lost session is rebuilt here (§10.4:
        RING up to keep + a kickoff STEPD with the same keep and anchor)."""
        box, self.box = self.box, None
        if box is None:
            return None
        t0 = time.perf_counter()
        since, rebuilds = None, 0
        while True:
            try:
                info, raw, timing = encoder.recv_stepd(idle)
                break
            except pipe_wire.BoxLost:
                raise
            except (OSError, RuntimeError, ValueError, struct.error, TypeError) as exc:
                if rebuilds >= pipe_wire.RESUME_TRIES:
                    encoder._drop()
                    raise pipe_wire.BoxLost(f'box refused STEPD at {box["keep"]} tokens after {rebuilds} rebuilt '
                                            f'sessions: {exc!r}'[:300]) from exc
                since = since or time.monotonic()

                def kickoff(enc):
                    if self.sticky is None and enc.dspark is not None:
                        self._ring(enc)
                        enc.send_stepd(box['keep'], box['anchor'], mode=dw.MODE_BOX, dmax=box['dmax'],
                                       flags=box['flags'])
                        count('steps')
                        count('kickoffs')
                        count('ring_resends')

                encoder.recover([box['anchor']], box['keep'], exc, since, resend=kickoff)
                rebuilds += 1
                if encoder._pending is None:
                    count('wait_ms', (time.perf_counter() - t0) * 1000.0)
                    return None  # today's path from here: the Mac drafts, _remote sends the plain STEP
        waited, rtt = (time.perf_counter() - t0) * 1000.0, timing.get('roundtrip_s', 0.0) * 1000.0
        count('wait_ms', waited)
        count('rtt_ms', rtt)
        self.times['wait'] += waited
        self.times['rtt'] += rtt
        self.times['box'] += info['box_s'] * 1000.0
        self.times['resolved'] += 1
        if info['mode_used'] == dw.MODE_BOX:
            encoder.hold_ready(info, raw, timing)
            if info['flags'] & dw.R_DRAFTED:
                count('cycles')
                count('drafted_tokens', info['L'] - 1)
            else:  # V2.3 filler ([anchor, anchor], the box could not draft): verified like any 1-draft block
                self.counts['filler'] += 1
                count('fillers')
                count('fallback_filler')
            return list(info['ids'][1:])
        # Bonus only (a v1 box, or the context end): an L=1 verify would leave the served L >= 2 path. Drop it.
        count('drained')
        count('fallback_bonus_only')
        return None

    def redraft(self, encoder, keep, ids):
        """After resolve() returned None: the Mac's block for (keep) as a kickoff STEPD mode 1 (nver = 0, same keep and
        anchor; the box ring is already current through keep). False: send a plain STEP (no session / no grant)."""
        self.counts['redraft'] += 1
        count('redrafts')
        if (self.sticky is not None or encoder.dspark is None or encoder.session is None or encoder._pending is not None
                or not 2 <= len(ids) <= 1 + dw.MAX_EXPLICIT):
            return False
        encoder.send_stepd(keep, ids[0], mode=dw.MODE_EXPLICIT, dmax=0, explicit=list(ids[1:]))
        count('steps')
        count('mode1')
        count('kickoffs')
        return True

    # ---- helpers -----------------------------------------------------------------------------------------------
    def _mac_ring(self, rebuild_fn, base):
        """Make the Mac GPU ring current through base (it was not appended while the box drafted)."""
        if not self.mac_current and self.ring.offset == base:
            rebuild_fn(*self.ring.window(), base)
            self.mac_current = True
            count('mac_rebuilds')

    def _plain(self, rebuild_fn, base):
        self._mac_ring(rebuild_fn, base)
        return 'plain', None

    def _ring(self, encoder):
        keys, taps = self.ring.window()
        rows = keys.shape[1] + taps.shape[0]
        if not rows:
            return
        try:
            nbytes = encoder.send_ring(self.ring.offset, keys.tobytes(), taps.tobytes(), keys.shape[1], taps.shape[0])
        except OSError:  # a lost link: the next receive rebuilds the session, which re-primes
            count('send_errors')
            return
        self.primed = True
        self.counts['rings'] += 1
        count('rings')
        count('ring_rows', rows)
        count('ring_bytes', nbytes)

    def _ring_first(self, encoder, verify, taps, base):
        """The box ring must hold [.., base) before a STEPD appends [base, keep); a kickoff carries no taps, so its
        RING covers [.., keep). While the box is off (DISABLED) it ignores RING: none is sent. Returns whether this
        cycle's taps are in the host ring already (a kickoff's are; otherwise the caller appends them)."""
        need = (not self.primed or verify is None) and not self.disabled
        if verify is None:
            self.ring.append(taps, base)
            if need:
                self._ring(encoder)
            return True
        if need:
            self._ring(encoder)
        return False

    def _ring_then_append(self, encoder, verify, taps, base):
        if not self._ring_first(encoder, verify, taps, base):
            self.ring.append(taps, base)

    def _send(self, encoder, keep, anchor, verify, taps, mode, explicit, dmax, flags, t0=None):
        nver, a, argmax, _ = verify if verify is not None else (0, 0, (), ())
        body = b''
        if nver:
            if self.disabled or flags & dw.F_NO_TAPS:
                flags |= dw.F_NO_TAPS  # the box ignores taps while off / PF_ACTIVE: a plain step
            else:
                body = taps.tobytes()
                count('tap_bytes', len(body))
        t1 = time.perf_counter()
        try:
            encoder.send_stepd(keep, anchor, nver=nver, a=a, mode=mode, dmax=dmax, flags=flags, argmax=argmax,
                               explicit=explicit, taps=body)
        except OSError:  # the step is recorded as in flight: its receive fails and rebuilds the session (§10.4)
            count('send_errors')
        if t0 is not None:
            t2 = time.perf_counter()
            count('send_ms', (t2 - t1) * 1000.0)
            count('prep_ms', (t2 - t0) * 1000.0)
            self.times['send'] += (t2 - t1) * 1000.0
            self.times['prep'] += (t2 - t0) * 1000.0
            self.times['sent'] += 1
        count('steps')
        count('mode0' if mode == dw.MODE_BOX else 'mode1')
        if not nver:
            count('kickoffs')

    def summary(self):
        t = self.times
        per = {k + '_ms': round(t[k] / t[n], 3) for k, n in (('prep', 'sent'), ('send', 'sent'), ('rtt', 'resolved'),
                                                              ('box', 'resolved'), ('wait', 'resolved')) if t[n]}
        return dict(self.counts, sticky=self.sticky, primed=self.primed, ring_offset=self.ring.offset, **per)
