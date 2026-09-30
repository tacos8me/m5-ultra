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
* Fused verify (og_fused.py): with three or more requests decoding, pairs of
  requests verify in one Mac pass (dense weights and head read once); the two
  pairs alternate, so the box runs one pair's steps while the Mac serves the
  other. Each request's arithmetic is bitwise its own forward_boundary.
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

from . import fast_encode, fe_trace, growth, og_cache, og_failover, og_fused, og_images, pipe_wire, spec_probe, woa_compact
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
             import_failed=0, import_fallbacks=0, kickoff_sent=0, kickoff_errors=0, fused_calls=0, fused_steps=0, draft_batches=0)
spec_probe.bind(STATS)  # retired copy-lock pre-send probe (DS41_OG_SPEC_PROBE=1 to re-enable): spec_probe_* keys
BOX_CACHE = os.environ.get('DS41_OG_BOX_CACHE', '1') == '1'
STATE = os.environ.get('DS41_OG_STATE', 'lean')
STREAM = os.environ.get('DS41_OG_STREAM', '1') == '1'
DELTA_MIN = int(os.environ.get('DS41_OG_DELTA_MIN', '4096'))
# Send the kickoff STEP (the last prompt token at N-1) as soon as the OPEN is done, so the box
# computes it while the Mac imports the state; the first forward finds it in flight (ensure_step).
KICKOFF = os.environ.get('DS41_OG_KICKOFF', '1') == '1'
# Wake the idle engine loop when a box OPEN completes (else the loop notices within step_interval, 50 ms).
WAKE = os.environ.get('DS41_OG_WAKE', '1') == '1'
STORE = og_cache.from_env()
# GPU keep-warm while a STEP is in flight: the box wait (~10 ms) is the one long GPU idle
# of a cycle, and the forward after it ran ~0.5 ms slower at 5 rows (partial-load bench,
# 10 ms idle with the host spinning; restored by this). A 4 MB add on a side stream every
# DS41_OG_GPU_WARM_US; never waited on, so the forward never queues behind it. 0 = off.
GPU_WARM_S = float(os.environ.get('DS41_OG_GPU_WARM_US', '500')) / 1e6
NUMERICS = [None]  # numerics key of the latest import; store lookups match it
# Fused multi-request verify (og_fused, DS41_OG_FUSE=0 = off): pair requests into one Mac pass once
# at least FUSE_MIN requests verify in a scheduler step. With 2 the lone pair waits for both box steps
# (per-request verify hides them), so pairs start at 3.
FUSE_MIN = int(os.environ.get('DS41_OG_FUSE_MIN', '3'))
# TTFT: emit the first token before the MTP post-init (its second forward is pre-sent to the box
# so it overlaps the emission), and build the copy-draft prompt index while the box prefills.
EARLY_FIRST = os.environ.get('DS41_OG_EARLY_FIRST', '1') == '1'
COPY_PREBUILD = os.environ.get('DS41_OG_COPY_PREBUILD', '1') == '1'
# Admission slices: while other requests decode, a new request's import+replay (the ~0.2-0.3 s tail replay
# of layers 20-39) runs in slices of about this much GPU time between their scheduler steps instead of in
# one piece (each stream then sees steps ~slice ms longer instead of one 0.2-0.3 s stall; the new request
# pays the interleaved steps). Same ops in the same order on the same stream. 0 = one piece (old behaviour).
ADMIT_SLICE_MS = float(os.environ.get('DS41_OG_ADMIT_SLICE_MS', '90'))
# The box refuses a STEP past its context length (ERR "context length exceeded", keep + rows > 1048576), and there
# is no local fallback: a request's prompt + output must stay below it, with room for one verify step's rows.
BOX_CONTEXT = int(os.environ.get('DS41_OG_BOX_CONTEXT', '1048576'))
STEP_MARGIN = 8
# Burst admission: start the box OPEN of up to this many text requests queued behind the head at once
# (OgPrefill.preopen) instead of one after another. 0 = head only (old behaviour).
PREOPEN = int(os.environ.get('DS41_OG_PREOPEN', '3'))
# chain (default): the next queued OPEN starts when every earlier one has finished on the box, so the box never
# prefills two of our prompts at once (the head's TTFT is unchanged; the Mac import and scheduler steps of
# request k overlap the box prefill of k+1). 0 = start them all at once (box interleaves the prefills: the whole
# burst starts together, the head's first token comes later).
PREOPEN_CHAIN = os.environ.get('DS41_OG_PREOPEN_CHAIN', '1') == '1'


def _layer_ms(rows):
    """Replay GPU time of one layer over `rows` tail rows (M5 Ultra, measured 24-128 rows): slice planning."""
    return 1.8 + 0.052 * rows
_lock = threading.Lock()
_ids = itertools.count(1)


def _fused_regime():
    """Fused-pair verify is live: enough open sessions for og_fused pairs (depth costs only)."""
    return og_fused.ENABLED and len(SESSIONS) >= FUSE_MIN


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
        if spec_probe.ENABLED:
            encoder = SESSIONS.get(sid)
            if encoder is not None:
                try:
                    spec_probe.close(encoder, request_id)
                except Exception:  # noqa: BLE001 -- statistics only
                    STATS['spec_probe_errors'] += 1
                    logger.debug('ds41-og spec probe close failed', exc_info=True)
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


DIGEST = os.environ.get('DS41_FE_DIGEST', '0') == '1'  # validation runs: log a digest of every imported state


def _log_digest(request_id, tokens, cache):
    """sha256 of the prompt ids and of every Mac-side cache array after the import (bitwise evidence)."""
    import hashlib
    import numpy as np
    h = hashlib.sha256(np.asarray(tokens, np.uint32).tobytes())
    ids = h.hexdigest()[:16]
    for i, item in enumerate(cache[20:], 20):
        for slot, x in enumerate(item.cache):
            if i == 20 and slot in (2, 3):
                continue  # layer 20's global rows are the box's bytes (+ stored delta base), not replayed
            if x is not None and x.size:
                h.update(np.array(x.view(mx.uint8) if x.dtype != mx.uint8 else x).tobytes())
    ctx = getattr(cache[0], '_omlx_mtp_prime_ctx', None)
    for stage in (ctx.caches if ctx is not None else ()):
        if stage.keys is not None:
            h.update(np.array(stage.keys.view(mx.uint8)).tobytes())
    logger.info('ds41-og digest %s: %d tokens, ids %s, state %s', request_id, len(tokens), ids, h.hexdigest()[:16])


def session_of(cache):
    value = cache[SID_LAYER][6]
    if value is None or value.shape != (1, 1):
        raise RuntimeError('ds41-og cache row carries no box session')
    sid = int(value.item())
    encoder = SESSIONS.get(sid)
    if encoder is None:
        raise RuntimeError(f'ds41-og box session {sid} is closed')
    return encoder


class GpuWarm:
    """Recv-spin hook of the engine thread: keeps the GPU clocked during the box wait."""

    def __init__(self, interval):
        self.interval, self.next, self.x, self.stream = interval, 0.0, None, None

    def __call__(self):
        now = time.perf_counter()
        if now < self.next:
            return
        self.next = now + self.interval
        if self.x is None:
            self.stream = mx.new_stream(mx.gpu)
            with mx.stream(self.stream):
                self.x = mx.zeros((1 << 20,), mx.float32)
                mx.eval(self.x)
        with mx.stream(self.stream):
            mx.async_eval(self.x + 1.0)


class OgLanguageModel(DecoderHalf):
    """Layers 20-39 + head + DSpark; layers 0-19 are remote placeholders."""

    _omlx_mtp_early_first = EARLY_FIRST

    def mtp_first_presend(self, gen_batch, token_id):
        """The first token went out before the post-init: send its forward's STEP now (ensure_step finds it)."""
        try:
            cache = gen_batch.prompt_cache
            encoder = session_of(cache)
            if encoder._pending is None:
                encoder.send_step([int(token_id)], cache[0].size())
                STATS['first_presend'] = STATS.get('first_presend', 0) + 1
        except Exception:  # noqa: BLE001 -- the post-init forward sends it itself
            STATS['presend_errors'] += 1
            logger.debug('ds41-og first presend skipped', exc_info=True)

    def mtp_copy_prompt_index(self, cache):
        """The request's prompt index built during its box OPEN (copy_draft.PromptIndex), once, or None."""
        try:
            prebuilt = session_of(cache).__dict__.pop('_og_copy_prompt', None)
        except RuntimeError:
            return None
        index = prebuilt.get() if prebuilt is not None else None
        STATS['copy_prebuilt' if index is not None else 'copy_local'] = STATS.get(
            'copy_prebuilt' if index is not None else 'copy_local', 0) + 1
        return index

    def set_tokenizer(self, tokenizer):
        # Engram hashing runs on the box.
        return None

    def set_token_map(self, token_map):
        return None

    def make_mtp_depth_controller(self, depth):
        return PipelineDepthController(depth, fused=_fused_regime)

    def _gpu_warm(self):
        if not GPU_WARM_S:
            return None
        warm = self.__dict__.get('_og_gpu_warm')
        if warm is None:
            warm = self.__dict__['_og_gpu_warm'] = GpuWarm(GPU_WARM_S)
        return warm

    def _remote(self, ids_list, caches, states_list):
        """Send every request's STEP first, then run layers 20-39 as replies land.

        A lost box session is rebuilt on its committed tokens and the step resent
        (pipe_wire.recover, bit-identical continuation); a box that stays down past
        DS41_OG_RESUME_WAIT_S raises BoxLost (og_failover turns it into resume markers).
        """
        sessions = [session_of(cache) for cache in caches]
        starts = [cache[0].size() for cache in caches]
        ids_list = [list(ids) for ids in ids_list]
        fuse = (len(caches) > 1 and all(states is not None for states in states_list)
                and og_fused.eligible(self, [len(ids) for ids in ids_list]))
        if spec_probe.ENABLED:
            _probe_steps(sessions, ids_list, starts)
        try:
            for encoder, ids, start in zip(sessions, ids_list, starts):
                STATS['present_hits'] += encoder.ensure_step_safe(ids, start)
            out, items = [], []
            for encoder, ids, start, cache, states in zip(sessions, ids_list, starts, caches, states_list):
                raw, timing = encoder.recv_step_safe(ids, start, self._gpu_warm())
                if spec_probe.ENABLED:
                    spec_probe.note_recv(encoder, start, timing)
                arrays = mlx_step(raw, len(ids))
                if fuse:
                    # One Mac pass for all of them once every boundary is in (og_fused).
                    items.append(dict(**arrays, cache=cache, start=start, verify_states=states))
                else:
                    logits, hidden = self.forward_boundary(
                        **arrays, cache=cache, start=start, verify=states is not None, verify_states=states)
                    cache[0]._pipe1_verify = None
                    out.append((logits, hidden))
                STATS['steps'] += 1
                STATS['rows'] += len(ids)
                STATS['box_s'] += timing['box_s']
                STATS['wait_s'] += timing['wait_s']
                STATS['roundtrip_s'] += timing['roundtrip_s']
                STATS['payload_s'] += timing['payload_s']
                trace = getattr(encoder, '_og_trace', None)
                if trace is not None and 'first_step' not in trace:
                    trace['first_step'] = time.time()
                    fe_trace.og_stamp(trace.get('request_id'), 'first_step', trace['first_step'])
                    _log_trace(trace)
                elif trace is not None and fe_trace.ON:
                    fe_trace.og_stamp(trace.get('request_id'), 'second_step')
            if fuse:
                out = self.forward_boundaries(items)
                for cache in caches:
                    cache[0]._pipe1_verify = None
                STATS['fused_calls'] += 1
                STATS['fused_steps'] += len(items)
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

    def mtp_verify_groups(self, lengths):
        """Verify groups for one scheduler step (fused_batch.advance), or None for per-request verify.

        With at least DS41_OG_FUSE_MIN requests to verify, consecutive requests of 2-5 rows pair
        up into one fused Mac pass (og_fused); a request of one row stays alone. Groups run one
        after another, each pre-sending its next STEPs after its drafts, so the box computes one
        pair's steps while the Mac verifies and drafts the other pair.
        """
        if not og_fused.ENABLED or len(lengths) < FUSE_MIN:
            return None
        groups, pair = [], []
        for i, length in enumerate(lengths):
            if 2 <= length <= 5:
                pair.append(i)
                if len(pair) == 2:
                    groups.append(pair)
                    pair = []
            else:
                groups.append([i])
        if pair:
            groups.append(pair)
        if all(len(g) == 1 for g in groups) or not og_fused.eligible(self, [2, 2]):
            return None
        return groups

    @property
    def mtp_draft_jobs_enabled(self):
        """Fused pairs draft together (dspark.proposal_forward_batch); DS41_OG_DRAFT_BATCH=0 = per request."""
        from . import dspark
        return dspark.DRAFT_BATCH

    def mtp_draft_jobs(self, jobs, depths):
        """Draft a verified group's next blocks (one DSpark pass when they share a width), then pre-send each."""
        from ..mlx_lm_mtp import batch_generator as bg
        STATS['draft_batches'] += bg.dspark_draft_jobs(jobs, depths)
        for gen_batch, state, *_ in jobs:
            try:
                presend(gen_batch, state)
            except Exception:  # noqa: BLE001 -- the ordinary send path still runs
                STATS['presend_errors'] += 1
                logger.debug('ds41-og presend skipped', exc_info=True)

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
        # No eval here: the next verify forward consumes these slices, so the
        # rollback costs no host sync (27a9b621; lost in the 45ef51f8 rebase).
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
    woa_compact.install(language_model)
    from . import dspark
    dspark.bind(STATS)  # draft_head_* keys in /og/stats
    dspark.install_draft_head(language_model)  # DS41_DRAFT_HEAD=mxfp8: quantize the draft head now
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
    tokenizer.has_tool_calling = True
    tokenizer.tool_call_start = tool_call_start
    tokenizer.tool_call_end = tool_call_end
    tokenizer.tool_parser = parse_tool_call
    if fast_encode.install(tokenizer) is not None:
        fast_encode.install_prepare_inputs(Processor)
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


class CopyPrompt(threading.Thread):
    """copy_draft.PromptIndex of a prompt, built beside the box OPEN (numpy drops the GIL for the heavy parts)."""

    def __init__(self, request_id, tokens):
        super().__init__(name=f'ds41-og-copy-{request_id}', daemon=True)
        self.tokens, self.index = tokens, None

    def run(self):
        from ..mlx_lm_mtp import copy_draft
        try:
            self.index = copy_draft.PromptIndex(self.tokens)
        except Exception:  # noqa: BLE001 -- the post-init builds the index itself
            logger.debug('ds41-og copy prompt index failed', exc_info=True)
        self.tokens = None

    def get(self):
        self.join()
        return self.index


class Job(threading.Thread):
    """Box OPEN (prefill + state transfer) off the engine thread.

    Retries while the box refuses connections (engine restarting) for up to
    DS41_OG_RESUME_WAIT_S, and a box that accepts but fails the open up to
    DS41_OG_RESUME_TRIES times.
    """

    def __init__(self, host, port, request_id, tokens, images=None, wake=None):
        super().__init__(name=f'ds41-og-open-{request_id}', daemon=True)
        self.host, self.port, self.request_id, self.tokens = host, port, request_id, list(tokens)
        self.wake = wake  # wakes the idle engine loop when the open is done (else it polls every 50 ms)
        self.images = list(images or ())
        # Prefix keys for the Mac row store: token ids, image-content keys inside image spans (og_images).
        self.keys = og_images.prompt_keys(self.tokens[:-1], self.images)
        self.encoder = self.result = self.error = self.base_rows = None
        self.copy_prompt = None
        self.done = threading.Event()
        self.cancelled = False
        self.cancel_event = threading.Event()
        # Admission slices (engine thread only): the import_state_steps generator, its result or error.
        self.replay = self.imported = self.replay_error = None
        self.replay_start = self.replay_marks = None

    def _open(self, delta_from):
        self.encoder = EncoderSession(self.host, self.port)
        if self.cancelled:  # abort() ran before this encoder existed: nothing interrupted it
            raise ConnectionAbortedError('request cancelled')
        self.result = open_remote(self.encoder, self.tokens, self.request_id,
                                  cache=BOX_CACHE, state=STATE, delta_from=delta_from, stream=STREAM,
                                  **({'images': self.images} if self.images else {}))
        return self.result[1]

    def _attempt(self):
        if self.cancelled:
            return
        self.base_rows = None
        entry, base = STORE.lookup(self.keys, NUMERICS[0]) if STORE is not None else (None, 0)
        if base < DELTA_MIN:
            entry, base = None, 0
        manifest = self._open(base)
        if self.cancelled:
            return
        delta = int(manifest.get('delta_from') or 0)
        if delta and (entry is None or delta != base or og_cache.numerics_key(manifest) != entry.key):
            # The box arithmetic changed since these rows were stored: never mix them.
            STATS['delta_retries'] += 1
            STORE.clear()
            self.encoder.close()
            if self.cancelled:
                return
            manifest = self._open(0)
            delta = int(manifest.get('delta_from') or 0)
            if delta:
                raise RuntimeError('box sent a delta state that was not requested')
        if delta:
            self.base_rows = (entry.kv, entry.index)
        info = self.encoder.open_info
        for key in ('t_ack', 't_first_part', 't_end'):
            if info.get(key) is not None:
                fe_trace.og_stamp(self.request_id, 'box_' + key[2:], info[key])
        if KICKOFF and not self.cancelled:
            try:
                self.encoder.send_step(self.tokens[-1:], len(self.tokens) - 1)
                STATS['kickoff_sent'] += 1
            except (OSError, ValueError) as exc:  # the first forward's ensure_step recovers the session
                STATS['kickoff_errors'] += 1
                logger.warning('ds41-og kickoff step for %s not sent: %r', self.request_id, exc)

    def run(self):
        since, failures = time.monotonic(), 0
        self.trace = dict(job_start=time.time())
        if COPY_PREBUILD and not self.cancelled and len(self.tokens) >= 64:
            from ..mlx_lm_mtp import copy_draft
            if copy_draft.ENABLED:
                self.copy_prompt = CopyPrompt(self.request_id, self.tokens)
                self.copy_prompt.start()
        try:
            while not self.cancelled:
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
                    self.cancel_event.wait(1.0)
        except BaseException as exc:  # noqa: BLE001 -- surfaced by should_defer()
            self.error = exc
            if self.encoder is not None:
                self.encoder.close()
        finally:
            if self.cancelled and self.encoder is not None:
                self.encoder.close()
            self.trace['job_end'] = time.time()
            fe_trace.og_stamp(self.request_id, 'job_start', self.trace['job_start'])
            fe_trace.og_stamp(self.request_id, 'job_end', self.trace['job_end'])
            self.done.set()
            if self.wake is not None:
                try:
                    self.wake()
                except Exception:  # noqa: BLE001 -- the engine loop still polls
                    logger.debug('ds41-og engine wake failed', exc_info=True)


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
            if not job.done.is_set():
                return True
            if fe_trace.ON and not getattr(job, 'seen_done', False):
                job.seen_done = True
                fe_trace.og_stamp(request.request_id, 'defer_done')
            if job.replay is not None or (ADMIT_SLICE_MS > 0 and job.imported is None and job.replay_error is None
                                          and getattr(scheduler, 'running', None)):
                return self.advance(scheduler, request, job)
            return False
        tokens = request.prompt_token_ids or []
        images = None
        if request.vlm_inputs_embeds is not None:
            images = og_images.REGISTRY.take(request.vlm_inputs_embeds)
            if images is None:
                og_failover.open_failed(request.request_id, ValueError(
                    'ds41-og: the image payload of this request is missing'), kind='invalid')
                return True
        room = BOX_CONTEXT - STEP_MARGIN - len(tokens)
        if len(tokens) < 2 or room < 1:
            # There is no local prefill: fail this request alone, cleanly.
            og_failover.open_failed(request.request_id, ValueError(
                f'ds41-og serves prompts of 2..{BOX_CONTEXT - STEP_MARGIN - 1} tokens (got {len(tokens)})'), kind='invalid')
            return True
        clamp_max_tokens(request, room)
        self.launch(scheduler, request, tokens, images)
        return True

    def launch(self, scheduler, request, tokens, images=None):
        job = Job(self.host, self.port, request.request_id, tokens, images, wake=opened_wake(scheduler) if WAKE else None)
        job.arrival = time.time() - (time.monotonic() - getattr(request, 'arrival_time', time.monotonic()))
        job.deferred = time.time()
        fe_trace.og_stamp(request.request_id, 'arrival', job.arrival)
        fe_trace.og_stamp(request.request_id, 'deferred', job.deferred)
        self.jobs[request.request_id] = job
        job.start()
        return job

    def preopen(self, scheduler):
        """Start the box OPEN of text requests queued behind the head of the waiting queue.

        The scheduler only offers waiting[0] to should_defer(), so without this a burst opens one
        request at a time: the next OPEN starts only after the previous request's OPEN and import
        (c4 at 8K: the 4th stream's first token ~3.1 s after arrival). Admission order is unchanged
        (FIFO, one request per step as before); only the network OPEN (box prefill + state transfer)
        starts early, within the scheduler's free slots, so at most max_num_seqs sessions exist.
        PREOPEN_CHAIN: one box prefill at a time, the next as soon as the previous OPEN is done.
        The imported state does not depend on when or how the OPEN ran (resume/delta are exact).
        """
        waiting = getattr(scheduler, 'waiting', None)
        if PREOPEN <= 0 or not waiting or len(waiting) < 2:
            return 0
        try:
            free = scheduler._effective_max_num_seqs() - scheduler._num_admitted_requests() - 1
        except Exception:  # noqa: BLE001 -- unknown scheduler shape: keep the old behaviour
            return 0
        head = self.jobs.get(waiting[0].request_id)
        ready = head is not None and head.done.is_set()  # chain: every earlier OPEN has finished
        budget, started = min(PREOPEN, free), 0
        for request in itertools.islice(waiting, 1, None):
            if budget <= 0:
                break
            budget -= 1
            job = self.jobs.get(request.request_id)
            if job is not None or request.request_id in self.failed:
                ready = ready and (job is None or job.done.is_set())
                continue
            tokens = request.prompt_token_ids or []
            if request.vlm_inputs_embeds is not None or len(tokens) < 2 or BOX_CONTEXT - STEP_MARGIN - len(tokens) < 1:
                ready = False  # images / invalid prompts keep the ordinary path at the head
                continue
            if PREOPEN_CHAIN and not ready:
                break
            clamp_max_tokens(request, BOX_CONTEXT - STEP_MARGIN - len(tokens))
            self.launch(scheduler, request, tokens)
            started += 1
            ready = False
        if started:
            STATS['preopened'] = STATS.get('preopened', 0) + started
        return started

    def advance(self, scheduler, request, job):
        """One admission slice of this request's import+replay (engine thread, engine stream).

        True while unfinished (the request stays deferred and the scheduler runs the decode step of the
        requests already running); False once imported or failed (prepare() then installs it or takes
        the failure path). With nobody else decoding, the rest runs at once.
        """
        lm = language_model_of(scheduler.model)
        budget = ADMIT_SLICE_MS if getattr(scheduler, 'running', None) else float('inf')
        with mx.stream(scheduler._stream):
            try:
                if job.replay is None:
                    tensors, manifest, _ = job.result
                    job.replay_start = time.perf_counter()
                    fe_trace.og_stamp(request.request_id, 'prepare_start')
                    marks = job.replay_marks = fe_trace.for_request(request.request_id)
                    job.replay = lm.import_state_steps(tensors, manifest, request.prompt_token_ids,
                                                       identity=job.encoder.identity, base_rows=job.base_rows,
                                                       **({'marks': marks} if marks is not None else {}))
                    STATS['admit_sliced'] = STATS.get('admit_sliced', 0) + 1
                spent = 0.0
                while spent < budget:
                    rows = next(job.replay)
                    spent += _layer_ms(rows or 128)
            except StopIteration as done:
                job.imported, job.replay = done.value, None
                return False
            except Exception as exc:  # noqa: BLE001 -- prepare() takes the failed-import path
                job.replay_error, job.replay = exc, None
                return False
        STATS['admit_slices'] = STATS.get('admit_slices', 0) + 1
        scheduler._ds41_opened = True  # a step that ran a slice has work: no idle wait before the next one
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
        start = time.perf_counter() if job.replay_start is None else job.replay_start
        if job.replay_start is None:
            fe_trace.og_stamp(request.request_id, 'prepare_start')
        marks = fe_trace.for_request(request.request_id)
        encoder = job.encoder
        lm = language_model_of(scheduler.model)
        try:
            while job.replay is not None:  # not expected: should_defer() finishes it; finish here if not
                try:
                    next(job.replay)
                except StopIteration as done:
                    job.imported, job.replay = done.value, None
            if job.replay_error is not None:
                raise job.replay_error
            if job.imported is not None:
                cache, rows = job.imported
            else:
                cache, rows = lm.import_state(tensors, manifest, tokens, identity=encoder.identity,
                                              base_rows=job.base_rows, **({'marks': marks} if marks is not None else {}))
            mx.eval([x for item in cache for x in item.cache if x is not None] + list(rows))
            fe_trace.og_stamp(request.request_id, 'import_eval')
            if DIGEST:
                _log_digest(request.request_id, tokens, cache)
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
        if job.copy_prompt is not None:
            encoder._og_copy_prompt = job.copy_prompt
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
        fe_trace.og_stamp(request.request_id, 'prepare_end')
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
            job.cancel_event.set()
            if not job.done.is_set() and job.encoder is not None:
                # An OPEN in flight (box prefill + state stream, up to ~60 s at 1M): cut it now instead of
                # letting the box finish a prefill nobody will use while other OPENs queue behind it.
                job.encoder.interrupt()
                STATS['open_cancelled'] = STATS.get('open_cancelled', 0) + 1

            def reap():
                job.done.wait()
                if job.encoder is not None:
                    job.encoder.close()
            threading.Thread(target=reap, daemon=True).start()
        close_request(request_id)


def clamp_max_tokens(request, room):
    """Cap this request's output at `room` tokens (prompt + output + one verify step within the box context):
    it then ends with finish_reason "length" instead of a STEP the box refuses."""
    import copy
    params = getattr(request, 'sampling_params', None)
    limit = getattr(params, 'max_tokens', None)
    if params is None or (isinstance(limit, int) and limit <= room):
        return
    params = copy.copy(params)  # never mutate a shared default
    params.max_tokens = room
    request.sampling_params = params
    STATS['max_tokens_clamped'] = STATS.get('max_tokens_clamped', 0) + 1
    logger.info('ds41-og %s: max_tokens %s -> %d (box context %d)', request.request_id, limit, room, BOX_CONTEXT)


def opened_wake(scheduler):
    """Called by a Job when its OPEN is done: mark the scheduler, then wake the idle engine loop.

    The mark covers an OPEN that finishes while a step is running (after its defer
    check): opened_step() then reports work, so the loop runs the admitting step at
    once instead of clearing the wake and sleeping step_interval.
    """
    def wake():
        scheduler._ds41_opened = True
        notify = getattr(scheduler, '_ds41_wake', None)
        if notify is not None:
            notify()
    return wake


def opened_step(step):
    """Scheduler.step wrapper: a step during which a box OPEN finished has work (see opened_wake)."""
    def wrapped(self):
        self._ds41_opened = False
        output = step(self)
        if getattr(self, '_ds41_opened', False) and output is not None and not getattr(output, 'has_work', True):
            output.has_work = True
        return output
    wrapped.__wrapped__ = step
    return wrapped


def _probe_steps(sessions, ids_list, starts):
    """Real STEPs of a forward that presend() did not see (first step, fallback paths): resolve predictions."""
    try:
        c1 = len(SESSIONS) == 1
        for encoder, ids, start in zip(sessions, ids_list, starts):
            spec_probe.observe_step(encoder, ids, start, None, c1)
    except Exception:  # noqa: BLE001 -- statistics only
        STATS['spec_probe_errors'] += 1
        logger.debug('ds41-og spec probe observe failed', exc_info=True)


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
    keep = cache[0].size()
    probe = spec_probe.ENABLED
    if probe:
        c1 = len(SESSIONS) == 1
        try:
            spec_probe.observe_step(encoder, ids, keep, getattr(state, "draft_source", None), c1)
        except Exception:  # noqa: BLE001 -- statistics only
            probe = False
            STATS['spec_probe_errors'] += 1
            logger.debug('ds41-og spec probe observe failed', exc_info=True)
    if encoder._pending is None:
        encoder.send_step(ids, keep)
    if probe:
        # After the send: the box computes this step meanwhile, so the prediction costs no c1 time.
        try:
            budget = int(gen_batch.max_tokens[0]) - int(gen_batch._num_tokens[0]) - len(state.queue) - 1
            spec_probe.predict(encoder, state, budget, keep, ids, c1)
        except Exception:  # noqa: BLE001 -- statistics only
            STATS['spec_probe_errors'] += 1
            logger.debug('ds41-og spec probe predict failed', exc_info=True)


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
    from omlx.engine_core import EngineCore
    engine_init = EngineCore.__init__

    def _engine_init(self, *args, **kwargs):
        engine_init(self, *args, **kwargs)
        if getattr(self, 'scheduler', None) is not None:
            self.scheduler._ds41_wake = self._wake_engine_loop

    EngineCore.__init__ = _engine_init
    defer, prepare = Scheduler._should_defer_for_cache_freshness, Scheduler._prepare_prefix_cache_for_request
    abort, cleanup = Scheduler._do_abort_request, Scheduler._cleanup_finished

    def _defer(self, request):
        try:
            if manager.should_defer(self, request):
                return True
            return defer(self, request)
        finally:
            # After the head's own OPEN was started: the box serves OPENs in arrival order.
            try:
                manager.preopen(self)
            except Exception:  # noqa: BLE001 -- the head-only path still admits everything
                logger.debug('ds41-og preopen skipped', exc_info=True)

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
    Scheduler.step = opened_step(Scheduler.step)
    Scheduler._prepare_prefix_cache_for_request = _prepare
    Scheduler._do_abort_request = _abort
    Scheduler._cleanup_finished = _cleanup
    og_failover.install(manager, close_request)
    logger.info('ds41-og installed: box %s:%d (resume wait %.0fs, %d tries)', host, int(port),
                pipe_wire.RESUME_WAIT_S, pipe_wire.RESUME_TRIES)
    return manager
