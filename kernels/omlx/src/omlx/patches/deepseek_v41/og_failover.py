# SPDX-License-Identifier: MIT
"""ds41-og worker-side failure handling (scheduler and API error paths).

Tier A (pipe_wire): a lost box session is rebuilt on the committed tokens and
the failed STEP is resent; nothing here is involved and nothing is visible to
the client beyond a pause.

When the box stays down past DS41_OG_RESUME_WAIT_S, `_remote` raises BoxLost.
It escapes scheduler.step(), so the engine loop fails every in-flight request.
This module records, for each of them, the output tokens the API layer has
already received, and appends them to the request's error as a resume marker
(`[ds41-og-resume:{json}]`, HTTP 503 for non-streaming). The CPU supervisor
strips the marker and continues the same client response on another backend
(tier B, og_serve/ds41_og.py + og_serve/og_resume.py).

A box OPEN that fails after the same wait fails only its own request, with an
empty resume marker (the supervisor replays it); before this, such a request
fell through to a local prefill that raised inside step() and failed every
in-flight request.
"""

import json
import logging
import threading
import time

logger = logging.getLogger(__name__)
MARK = 'ds41-og-resume:'
LOST = dict(at=0.0, reason='')
RESUMABLE = {}   # request_id -> resume info, consumed by the API error path
_failed_opens = {}  # request_id -> (kind, reason), failed by the next step()
_emitted = set()  # failed opens already answered (until their abort cleanup runs)
_lock = threading.Lock()


def note_lost(exc):
    LOST.update(at=time.monotonic(), reason=str(exc)[:300])


def open_failed(request_id, exc, kind='box'):
    with _lock:
        if request_id not in _emitted:
            _failed_opens.setdefault(request_id, (kind, f'{type(exc).__name__}: {exc}'[:300]))


def clear_open(request_id):
    with _lock:
        _failed_opens.pop(request_id, None)
        _emitted.discard(request_id)


def _remember(request_id, info):
    now = time.monotonic()
    for key in [k for k, v in RESUMABLE.items() if now - v['t'] > 600]:
        RESUMABLE.pop(key, None)  # never consumed (client gone)
    RESUMABLE[request_id] = dict(info, t=now)


def import_failed(request, exc):
    """The box state could not be imported, even from a plain full OPEN: fail this request alone.

    It carries an empty resume marker (the supervisor replays it on another backend) and never
    reaches a local prefill (og has none; see guard_local_prefill)."""
    reason = f'ds41-og: box state import failed ({type(exc).__name__}: {exc})'[:300]
    _remember(request.request_id, dict(kind='box', reason=reason, output=[],
                                       prompt_tokens=int(request.num_prompt_tokens or 0)))
    request._ds41_og_failed = reason


def guard_local_prefill(Scheduler):
    """og has no local prefill: a request whose box import failed errors out before any forward."""
    external, chunk = Scheduler._do_external_prefill, Scheduler._step_prefill_chunk

    def _do_external_prefill(self, request, *args, **kwargs):
        reason = getattr(request, '_ds41_og_failed', None)
        if reason:
            raise RuntimeError(reason)
        return external(self, request, *args, **kwargs)

    def _step_prefill_chunk(self, state, *args, **kwargs):
        reason = getattr(getattr(state, 'request', None), '_ds41_og_failed', None)
        if reason:
            raise RuntimeError(reason)
        return chunk(self, state, *args, **kwargs)

    Scheduler._do_external_prefill = _do_external_prefill
    Scheduler._step_prefill_chunk = _step_prefill_chunk


def _take_failed_opens():
    with _lock:
        items = list(_failed_opens.items())
        _failed_opens.clear()
        _emitted.update(request_id for request_id, _ in items)
    return items


def install(manager, close_request):
    from omlx import engine_core
    from omlx.request import RequestOutput
    from omlx.scheduler import Scheduler

    fail_all, step = Scheduler.fail_all_requests, Scheduler.step

    def _fail_all(self):
        box = time.monotonic() - LOST['at'] < 10
        infos = {}
        for request_id, request in list(self.requests.items()):
            infos[request_id] = dict(kind='box' if box else 'engine', reason=LOST['reason'] if box else 'engine error',
                                     prompt_tokens=int(request.num_prompt_tokens or 0),
                                     output=[int(t) for t in (request.output_token_ids or [])])
        failed = fail_all(self)
        for request_id in failed:
            info = infos.get(request_id)
            if info is not None and info['kind'] == 'box':
                _remember(request_id, info)
            # Box sessions of failed requests would otherwise stay open on the box.
            manager.abort(request_id)
            close_request(request_id)
        if box:
            logger.error('ds41-og box lost: failed %d request(s) with resume markers', len(failed))
        return failed

    def _step(self):
        out = step(self)
        for request_id, (kind, reason) in _take_failed_opens():
            request = self.requests.get(request_id)
            if request is None:
                continue
            message = f'ds41-og: the RTX box could not prefill this request ({reason})'
            if kind == 'box':
                _remember(request_id, dict(kind='box', reason=reason, output=[],
                                           prompt_tokens=int(request.num_prompt_tokens or 0)))
            self.abort_request(request_id)
            out.outputs.append(RequestOutput(request_id=request_id, finished=True, finish_reason='error',
                                             error=message, error_code='box_unavailable'))
            out.finished_request_ids.add(request_id)
            out.has_work = True
            logger.error('ds41-og %s: %s', request_id, message)
        return out

    raise_error = engine_core._raise_request_output_error

    def _raise(output):
        info = RESUMABLE.pop(output.request_id, None)
        if info is None:
            return raise_error(output)
        info.pop('t', None)
        from fastapi import HTTPException
        detail = f'{output.error or "ds41-og: box unavailable"} [{MARK}{json.dumps(info, separators=(",", ":"))}]'
        raise HTTPException(status_code=503, detail=detail, headers={'Retry-After': '30'})

    Scheduler.fail_all_requests = _fail_all
    Scheduler.step = _step
    guard_local_prefill(Scheduler)
    engine_core._raise_request_output_error = _raise
