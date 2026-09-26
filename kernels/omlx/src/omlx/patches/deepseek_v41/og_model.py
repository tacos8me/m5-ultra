# SPDX-License-Identifier: MIT
"""ds41-og: the original-weight split pipeline behind omlx's unchanged scheduler.

The RTX box owns layers 0-19 (and the layer-20 KV/index rows) of every request
in a per-request TCP session (STEP_API.md); this process owns layers 20-39,
the head and DSpark. The omlx scheduler, DSpark/copy/tool draft loop, EVICT
cost policy, sampling, tool/reasoning parsing and OpenAI API run unchanged:

* Prefill: every request is deferred while a network thread OPENs its box
  session (the box prefills tokens[:-1] and returns the encoder state); the
  engine thread imports and replays it as an exact prefix-cache hit covering
  tokens[:-1], exactly like split-wire's phase-2 hook. There is no local
  prefill: a failed box open fails that request with a clean error.
* Decode/verify: each forward sends the rows to the box (rollback is implied by
  `keep` = the cache offset) and runs layers 20-39 on the returned boundary.
  Two requests' STEPs are sent before either reply is read.
* Session binding: layer 1's Engram-history slot (unused on the Mac, Engram runs
  on the box) holds the session id as a [1, 1] int64 row, so it follows every
  cache extract/merge/extend/filter the scheduler performs for batching.
* Prefix reuse (og_cache.py): OPEN asks the box to resume from its exact
  snapshots (``cache``) and to send layer 20 + tail only (``lean``); the Mac
  keeps layer 20's global rows per prefix and requests only the rows past the
  longest stored common prefix (``delta_from``). The imported state is the
  one a fresh prefill of the same prompt yields.
"""

import itertools
import json
import logging
import os
from pathlib import Path
import threading
import time

import mlx.core as mx
import mlx.nn as nn

from . import growth, og_cache, og_failover, og_images, pipe_wire
from .cache import DeepseekV41Cache
from .pipe_decoder import DecoderHalf, load_decoder
from .pipe_session import PipelineDepthController, open_remote
from .pipe_wire import BoxLost, EncoderSession, mlx_state, mlx_step

logger = logging.getLogger(__name__)
SID_LAYER = 1
SESSIONS = {}
REQUESTS = {}
STATS = dict(opened=0, open_failed=0, open_retries=0, closed=0, steps=0, box_s=0.0, wait_s=0.0, rows=0, roundtrip_s=0.0,
             payload_s=0.0, single_calls=0, multi_calls=0, present_hits=0, presend_errors=0, box_lost=0,
             box_resumed=0, box_resumed_tokens=0, delta_opens=0, delta_rows=0, delta_retries=0,
             import_failed=0, import_fallbacks=0)
BOX_CACHE = os.environ.get('DS41_OG_BOX_CACHE', '1') == '1'
STATE = os.environ.get('DS41_OG_STATE', 'lean')
STREAM = os.environ.get('DS41_OG_STREAM', '1') == '1'
DELTA_MIN = int(os.environ.get('DS41_OG_DELTA_MIN', '4096'))
STORE = og_cache.from_env()
NUMERICS = [None]  # numerics key of the latest import; store lookups match it
_lock = threading.Lock()
_ids = itertools.count(1)


def register(encoder):
    sid = next(_ids)
    with _lock:
        SESSIONS[sid] = encoder
    return sid


def close_session(sid):
    with _lock:
        encoder = SESSIONS.pop(sid, None)
    if encoder is not None:
        encoder.close()
        STATS['closed'] += 1


def close_request(request_id):
    sid = REQUESTS.pop(request_id, None)
    if sid is not None:
        close_session(sid)


def _log_trace(t):
    """One line per request: where the admission time went (wall clock, seconds)."""
    def gap(a, b):
        return round(t[b] - t[a], 3) if t.get(a) is not None and t.get(b) is not None else None
    logger.info('ds41-og admission %s: %d tokens, arrival->defer %s, defer->job %s, box open %s, job->prepare %s, '
                'import %s, prepare->first step %s, arrival->first step %s (first_step_wall %.3f)',
                t['request_id'], t['tokens'], gap('arrival', 'deferred'), gap('deferred', 'job_start'),
                gap('job_start', 'job_end'), gap('job_end', 'prepare_start'), gap('prepare_start', 'prepare_end'),
                gap('prepare_end', 'first_step'), gap('arrival', 'first_step'), t['first_step'])


def session_of(cache):
    value = cache[SID_LAYER][6]
    if value is None or value.shape != (1, 1):
        raise RuntimeError('ds41-og cache row carries no box session')
    sid = int(value.item())
    encoder = SESSIONS.get(sid)
    if encoder is None:
        raise RuntimeError(f'ds41-og box session {sid} is closed')
    return encoder


class OgLanguageModel(DecoderHalf):
    """Layers 20-39 + head + DSpark; layers 0-19 are remote placeholders."""

    def set_tokenizer(self, tokenizer):
        # Engram hashing runs on the box.
        return None

    def set_token_map(self, token_map):
        return None

    def make_mtp_depth_controller(self, depth):
        return PipelineDepthController(depth)

    def _remote(self, ids_list, caches, states_list):
        """Send every request's STEP first, then run layers 20-39 as replies land.

        A lost box session is rebuilt on its committed tokens and the step resent
        (pipe_wire.recover, bit-identical continuation); a box that stays down past
        DS41_OG_RESUME_WAIT_S raises BoxLost (og_failover turns it into resume markers).
        """
        sessions = [session_of(cache) for cache in caches]
        starts = [cache[0].size() for cache in caches]
        ids_list = [list(ids) for ids in ids_list]
        try:
            for encoder, ids, start in zip(sessions, ids_list, starts):
                STATS['present_hits'] += encoder.ensure_step_safe(ids, start)
            out = []
            for encoder, ids, start, cache, states in zip(sessions, ids_list, starts, caches, states_list):
                raw, timing = encoder.recv_step_safe(ids, start)
                arrays = mlx_step(raw, len(ids))
                logits, hidden = self.forward_boundary(
                    **arrays, cache=cache, start=start, verify=states is not None, verify_states=states)
                cache[0]._pipe1_verify = None
                STATS['steps'] += 1
                STATS['rows'] += len(ids)
                STATS['box_s'] += timing['box_s']
                STATS['wait_s'] += timing['wait_s']
                STATS['roundtrip_s'] += timing['roundtrip_s']
                STATS['payload_s'] += timing['payload_s']
                out.append((logits, hidden))
                trace = getattr(encoder, '_og_trace', None)
                if trace is not None and 'first_step' not in trace:
                    trace['first_step'] = time.time()
                    _log_trace(trace)
        except BoxLost as exc:
            STATS['box_lost'] += 1
            og_failover.note_lost(exc)
            raise
        return out

    def _forward(self, input_ids, cache=None, inputs_embeds=None, token_types=None, **kwargs):
        if inputs_embeds is not None or kwargs.get('_ced_prefill', False) or cache is None:
            raise RuntimeError('ds41-og prefills on the RTX box only (text requests; box open failed?)')
        capture = bool(kwargs.get('return_dspark_hidden', False))
        states = kwargs.get('mtp_verify_states')
        batch, length = input_ids.shape
        STATS['single_calls'] += 1
        if batch == 1:
            logits, hidden = self._remote([input_ids[0].tolist()], [cache], [states])[0]
            for item in cache:
                item.advance(length)
            return (logits, hidden) if capture else logits
        # Batched plain decode (scheduler fallback paths): one session per row.
        if capture or states is not None:
            raise ValueError('ds41-og batched rows support plain decode only')
        masks = cache[0].make_mask(length)
        if masks is not None and not bool(mx.all(masks).item()):
            raise ValueError('ds41-og does not support padded decode rows')
        rows = [[item.extract(r) for item in cache] for r in range(batch)]
        outs = self._remote([input_ids[r].tolist() for r in range(batch)], rows, [None] * batch)
        for i, item in enumerate(cache):
            item.adopt(DeepseekV41Cache.merge([row[i] for row in rows]))
            item.advance(length)
        return mx.concatenate([logits for logits, _ in outs], 0)

    def mtp_verify_requests(self, inputs, caches):
        if not inputs or len(inputs) != len(caches):
            raise ValueError('DSpark verify needs one cache per request')
        STATS['multi_calls'] += 1
        snapshots = [[(list(item.cache), item.left_padding, item.lengths) for item in cache] for cache in caches]
        starts = [cache[0].size() for cache in caches]
        states = [[{} for _ in cache] for cache in caches]
        outs = self._remote([ids[0].tolist() for ids in inputs], caches, states)
        results = []
        for ids, cache, snapshot, start, state, (logits, hidden) in zip(inputs, caches, snapshots, starts, states, outs):
            for item in cache:
                item.advance(ids.shape[1])
            cache[0]._mtp_draft_stash = (ids, snapshot, start, state)
            results.append((logits, hidden, None))
        return results

    def mtp_partial_rollback(self, cache, accepted, num_drafts):
        """Served rollback semantics on layers 20-39; the box rewinds on the next STEP."""
        if not 0 <= accepted <= num_drafts:
            return False
        stash = getattr(cache[0], '_mtp_draft_stash', None)
        if stash is None:
            return accepted == num_drafts
        inputs, snapshots, before, verify_states = stash
        if inputs.shape[1] != num_drafts + 1:
            return False
        if any(item.size() != before + inputs.shape[1] for item in cache):
            return False
        cache[0]._mtp_draft_stash = None
        if accepted == num_drafts:
            return True
        count = accepted + 1
        end = before + count
        window = self._config.window_size
        for i, (item, snapshot, state) in enumerate(zip(cache, snapshots, verify_states)):
            if i >= 20:
                window_end = min(before, window) + count
                item[1] = state['window'][:, max(0, window_end - window):window_end]
                if item.compress_ratio:
                    if item.compress_ratio != 1:
                        raise ValueError('ds41-og decoder half only has the ratio-1 layer-20 cache')
                    growth.truncate(item, 2, end)
                    growth.truncate(item, 3, end)
            item[0] = mx.array([end], mx.int32)
            item.left_padding, item.lengths = snapshot[1], snapshot[2]
            item.advance(count)
        mx.eval([item.state for item in cache[20:]])
        return True


class OgModel(nn.Module):
    """Model-shaped wrapper for omlx's VLM adapter. Images: the box runs the vision tower (og_images)."""

    def __init__(self, language_model):
        super().__init__()
        self.config = language_model._config
        self.model_type = self.config.model_type
        self.language_model = language_model

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=self.language_model.embed(input_ids))
        if input_ids.shape[0] != 1:
            raise ValueError('Prepare image embeddings per request')
        images = og_images.build(input_ids, pixel_values, self.config.image_token_id,
                                 kwargs['image_grids'], kwargs['image_spans'], kwargs['image_types'])
        # No Mac-side embeddings are needed (the box prefills the prompt, images included); the placeholder only
        # carries the images to the admission hook (OgPrefill.should_defer).
        placeholder = mx.zeros((1, input_ids.shape[1], 1), mx.bfloat16)
        og_images.REGISTRY.put(placeholder, images)
        return InputEmbeddingsFeatures(inputs_embeds=placeholder)

    def __call__(self, input_ids, pixel_values=None, cache=None, **kwargs):
        if pixel_values is not None:
            raise ValueError('ds41-og prefills images on the box, never in a local forward')
        return self.language_model(input_ids, cache=cache, **{
            k: kwargs[k] for k in ('return_hidden', 'return_dspark_hidden', 'n_confirmed') if k in kwargs})

    def prefetch_ple(self, next_ids, current_ids):
        return None

    def close(self):
        return None


def load(path, **kwargs):
    """Loader for pipe1 decoder-half checkpoints (patched over loading.load)."""
    from transformers import PreTrainedTokenizerFast
    from .processing import Processor
    from .tool_parser import parse_tool_call, tool_call_end, tool_call_start

    language_model = load_decoder(path, cls=OgLanguageModel)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
    tokenizer.has_tool_calling = True
    tokenizer.tool_call_start = tool_call_start
    tokenizer.tool_call_end = tool_call_end
    tokenizer.tool_parser = parse_tool_call
    model = OgModel(language_model)
    return model, Processor(tokenizer, model.config)


def language_model_of(model):
    seen = 0
    while not (hasattr(model, 'layers') and hasattr(model, '_config')) and seen < 4:
        inner = getattr(model, '_language_model', None) or getattr(model, 'language_model', None)
        if inner is None:
            break
        model, seen = inner, seen + 1
    return model


class Job(threading.Thread):
    """Box OPEN (prefill + state transfer) off the engine thread.

    Retries while the box refuses connections (engine restarting) for up to
    DS41_OG_RESUME_WAIT_S, and a box that accepts but fails the open up to
    DS41_OG_RESUME_TRIES times.
    """

    def __init__(self, host, port, request_id, tokens, images=None):
        super().__init__(name=f'ds41-og-open-{request_id}', daemon=True)
        self.host, self.port, self.request_id, self.tokens = host, port, request_id, list(tokens)
        self.images = list(images or ())
        # Prefix keys for the Mac row store: token ids, image-content keys inside image spans (og_images).
        self.keys = og_images.prompt_keys(self.tokens[:-1], self.images)
        self.encoder = self.result = self.error = self.base_rows = None
        self.done = threading.Event()
        self.cancelled = False

    def _open(self, delta_from):
        self.encoder = EncoderSession(self.host, self.port)
        self.result = open_remote(self.encoder, self.tokens, self.request_id,
                                  cache=BOX_CACHE, state=STATE, delta_from=delta_from, stream=STREAM,
                                  **({'images': self.images} if self.images else {}))
        return self.result[1]

    def _attempt(self):
        self.base_rows = None
        entry, base = STORE.lookup(self.keys, NUMERICS[0]) if STORE is not None else (None, 0)
        if base < DELTA_MIN:
            entry, base = None, 0
        manifest = self._open(base)
        delta = int(manifest.get('delta_from') or 0)
        if delta and (entry is None or delta != base or og_cache.numerics_key(manifest) != entry.key):
            # The box arithmetic changed since these rows were stored: never mix them.
            STATS['delta_retries'] += 1
            STORE.clear()
            self.encoder.close()
            manifest = self._open(0)
            delta = int(manifest.get('delta_from') or 0)
            if delta:
                raise RuntimeError('box sent a delta state that was not requested')
        if delta:
            self.base_rows = (entry.kv, entry.index)

    def run(self):
        since, failures = time.monotonic(), 0
        self.trace = dict(job_start=time.time())
        try:
            while True:
                try:
                    self._attempt()
                    return
                except (OSError, RuntimeError, ValueError) as exc:
                    busy = isinstance(exc, pipe_wire.BoxBusy)
                    if self.encoder is not None:
                        self.encoder.close()
                        self.encoder = None
                        failures += not busy
                    if (self.cancelled or failures >= pipe_wire.RESUME_TRIES
                            or time.monotonic() - since >= pipe_wire.wait_limit(exc)):
                        raise
                    STATS['open_retries'] += 1
                    logger.warning('ds41-og box open for %s failed (%r); retrying', self.request_id, exc)
                    time.sleep(1.0)
        except BaseException as exc:  # noqa: BLE001 -- surfaced by should_defer()
            self.error = exc
            if self.encoder is not None:
                self.encoder.close()
        finally:
            self.trace['job_end'] = time.time()
            self.done.set()


class OgPrefill:
    def __init__(self, host, port):
        self.host, self.port = host, int(port)
        self.jobs = {}
        self.failed = set()

    def should_defer(self, scheduler, request):
        job = self.jobs.get(request.request_id)
        if job is not None:
            if job.done.is_set() and job.error is not None:
                # Keep it out of the local prefill path; og_failover fails it in step().
                STATS['open_failed'] += 1 if request.request_id not in self.failed else 0
                self.failed.add(request.request_id)
                og_failover.open_failed(request.request_id, job.error)
                return True
            return not job.done.is_set()
        tokens = request.prompt_token_ids or []
        images = None
        if request.vlm_inputs_embeds is not None:
            images = og_images.REGISTRY.take(request.vlm_inputs_embeds)
            if images is None:
                og_failover.open_failed(request.request_id, ValueError(
                    'ds41-og: the image payload of this request is missing'), kind='invalid')
                return True
        if not 2 <= len(tokens) <= 1048576:
            # There is no local prefill: fail this request alone, cleanly.
            og_failover.open_failed(request.request_id, ValueError(
                f'ds41-og serves prompts of 2..1048576 tokens (got {len(tokens)})'), kind='invalid')
            return True
        job = Job(self.host, self.port, request.request_id, tokens, images)
        job.arrival = time.time() - (time.monotonic() - getattr(request, 'arrival_time', time.monotonic()))
        job.deferred = time.time()
        self.jobs[request.request_id] = job
        job.start()
        return True

    def prepare(self, scheduler, request):
        job = self.jobs.pop(request.request_id, None)
        if job is None:
            return False
        job.done.wait()
        tokens = request.prompt_token_ids
        if job.error is not None:  # not reached: should_defer() keeps failed opens waiting
            STATS['open_failed'] += 1
            logger.error('ds41-og box open failed for %s: %s', request.request_id, job.error)
            return False
        tensors, manifest, open_s = job.result
        start = time.perf_counter()
        encoder = job.encoder
        lm = language_model_of(scheduler.model)
        try:
            cache, rows = lm.import_state(tensors, manifest, tokens, identity=encoder.identity,
                                          base_rows=job.base_rows)
            mx.eval([x for item in cache for x in item.cache if x is not None] + list(rows))
            if STORE is not None:
                NUMERICS[0] = og_cache.numerics_key(manifest)
                STORE.record(job.keys, *rows, NUMERICS[0])
        except Exception:
            encoder.close()
            STATS['import_failed'] += 1
            logger.exception('ds41-og import failed for %s; retrying with a plain full OPEN', request.request_id)
            if STORE is not None:
                STORE.clear()  # never reuse rows around a delta that did not import
            try:
                encoder, cache, manifest, open_s = self.plain_open(lm, request, tokens, job.images)
            except Exception as exc:  # noqa: BLE001 -- fails this request alone, never a local prefill
                STATS['open_failed'] += 1
                logger.exception('ds41-og plain OPEN after a failed import failed for %s', request.request_id)
                og_failover.import_failed(request, exc)
                return False
            STATS['import_fallbacks'] += 1
        sid = register(encoder)
        cache[SID_LAYER][6] = mx.array([[sid]], mx.int64)
        mx.eval(cache[SID_LAYER][6])
        REQUESTS[request.request_id] = sid
        encoder._og_trace = dict(job.trace, arrival=getattr(job, 'arrival', None), deferred=getattr(job, 'deferred', None),
                                     prepare_start=start + (time.time() - time.perf_counter()), prepare_end=time.time(),
                                     request_id=request.request_id, tokens=len(tokens))
        STATS['opened'] += 1
        request._ds41_remote_imported = True
        request.prompt_cache = cache
        request.cached_tokens = len(tokens) - 1
        request.remaining_tokens = tokens[-1:]
        scheduler._prefix_cache_prepared.add(request.request_id)
        info = encoder.open_info
        resumed = info.get('resumed_tokens') or 0
        delta = int(manifest.get('delta_from') or 0)
        STATS['box_resumed'] += bool(resumed)
        STATS['box_resumed_tokens'] += resumed
        STATS['delta_opens'] += bool(delta)
        STATS['delta_rows'] += delta
        STATS['image_opens'] = STATS.get('image_opens', 0) + bool(job.images)
        logger.info('ds41-og %s: %d tokens, %d image(s), box open %.2fs (resumed %d, prefill %.2fs, %d bytes, delta_from %d), '
                    'import+replay %.2fs, session %d',
                    request.request_id, len(tokens), len(job.images), open_s, resumed, info.get('box_prefill_s') or 0,
                    info.get('state_bytes') or 0, delta, time.perf_counter() - start, sid)
        return True

    def plain_open(self, lm, request, tokens, images=None):
        """Full OPEN without the box cache or a delta, imported by the pre-cache path (import_prefill)."""
        encoder = EncoderSession(self.host, self.port)
        try:
            tensors, manifest, open_s = open_remote(encoder, tokens, request.request_id,
                                                    **({'images': images} if images else {}))
            cache = lm.import_prefill(mlx_state(tensors), manifest, tokens, identity=encoder.identity)
            mx.eval([x for item in cache for x in item.cache if x is not None])
        except BaseException:
            encoder.close()
            raise
        return encoder, cache, manifest, open_s

    def abort(self, request_id):
        og_failover.clear_open(request_id)
        self.failed.discard(request_id)
        job = self.jobs.pop(request_id, None)
        if job is not None:
            job.cancelled = True

            def reap():
                job.done.wait()
                if job.encoder is not None:
                    job.encoder.close()
            threading.Thread(target=reap, daemon=True).start()
        close_request(request_id)


def presend(gen_batch, state):
    """After a request's accept+draft, send its next verify rows to the box at once.

    The next cycle's forward finds the identical STEP in flight (ensure_step), so
    the box computes it while the Mac verifies/drafts the other request (c2) or
    while the scheduler finishes this cycle (c1). A step that is not used (the
    request ended, or the loop issues different rows) is consumed or dropped
    with its session; STEP's `keep` rewinds the box either way.
    """
    if state.next_main is None or state.drafts is None or state.queue is None:
        return
    host = language_model_of(gen_batch.model)
    if not isinstance(host, OgLanguageModel):
        return
    cache = gen_batch.prompt_cache
    ids = [int(x) for x in state.next_main.tolist()] + [int(x) for x in state.drafts.tolist()]
    if not 1 <= len(ids) <= 5:
        return
    encoder = session_of(cache)
    if encoder._pending is None:
        encoder.send_step(ids, cache[0].size())


def install(host='10.10.10.1', port=10052):
    """Patch the loader and scheduler hooks; returns the prefill manager."""
    from . import loading
    from omlx.scheduler import Scheduler

    original = loading.load

    def og_load(path, **kwargs):
        raw = json.loads((Path(path) / 'config.json').read_text())
        if 'pipe1' not in raw:
            return original(path, **kwargs)
        return load(path, **kwargs)

    loading.load = og_load
    manager = OgPrefill(host, port)
    defer, prepare = Scheduler._should_defer_for_cache_freshness, Scheduler._prepare_prefix_cache_for_request
    abort, cleanup = Scheduler._do_abort_request, Scheduler._cleanup_finished

    def _defer(self, request):
        if manager.should_defer(self, request):
            return True
        return defer(self, request)

    def _prepare(self, request):
        if request.request_id in self._prefix_cache_prepared:
            return
        if manager.prepare(self, request):
            return
        prepare(self, request)

    def _abort(self, request_id):
        manager.abort(request_id)
        return abort(self, request_id)

    def _cleanup(self, finished_ids):
        try:
            return cleanup(self, finished_ids)
        finally:
            for request_id in list(finished_ids):
                close_request(request_id)

    from ..mlx_lm_mtp import batch_generator as bg
    chain = bg._run_verify_cycle_chain

    def _chain(gen_batch, state, *args, **kwargs):
        result = chain(gen_batch, state, *args, **kwargs)
        if not kwargs.get('defer_commit') and kwargs.get('draft_jobs') is None:
            try:
                presend(gen_batch, state)
            except Exception:  # noqa: BLE001 -- the ordinary send path still runs
                STATS['presend_errors'] += 1
                logger.debug('ds41-og presend skipped', exc_info=True)
        return result

    bg._run_verify_cycle_chain = _chain
    Scheduler._should_defer_for_cache_freshness = _defer
    Scheduler._prepare_prefix_cache_for_request = _prepare
    Scheduler._do_abort_request = _abort
    Scheduler._cleanup_finished = _cleanup
    og_failover.install(manager, close_request)
    logger.info('ds41-og installed: box %s:%d (resume wait %.0fs, %d tries)', host, int(port),
                pipe_wire.RESUME_WAIT_S, pipe_wire.RESUME_TRIES)
    return manager
