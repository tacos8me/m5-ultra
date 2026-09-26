# split-nv phase 3 step API (draft v1) — box decode half, 10.10.10.1:10052, plain TCP

Envelope = split-wire's `struct '<4sIQ'` (tag, header_bytes, payload_bytes) so pipe1 can reuse `ds41_wire.recv_header/recv_into`.
Control tags carry a JSON header; hot-path tags (STEP/STPR) carry a fixed packed header and no JSON.
One TCP connection = one session (close the socket = free the session). Little-endian throughout.

| tag  | dir      | header                                                                 | payload |
|------|----------|------------------------------------------------------------------------|---------|
| OPEN | Mac->box | JSON {proto:1, identity, prompt_tokens:N, token_sha256, state:"full"\|"lean"\|"none", request_id, stream?:1, cache?:1, delta_from?:P, prefix_sha256?, images?:[{start, grid:[vit_h, vit_w], sha256, length?}]} | uint32[N] token ids (then, with images, each image's bf16 patches in order) |
| ACK  | box->Mac | JSON {ok, session, prefill_s, resumed_tokens} (stream: sent before the prefill, {ok, session, stream:1, resumed_tokens, est_s}) | - |
| STAT | box->Mac | JSON {format:"ds41-encoder-state-v1", prompt_tokens, bytes, prefill_s} (omitted when OPEN.state=="none") | the complete `ds41-encoder-state-v1` safetensors file (identical bytes to `POST /v1/prefill`), one frame |
| STEP | Mac->box | packed `<IIH` (session, keep, L)  — first roll the session back to `keep` prompt+accepted tokens (keep <= current length), then append the L token ids at positions [keep, keep+L) and run layers 0-19 (+ layer-20 KV/index projection) for them | uint32[L] token ids |
| STPR | box->Mac | packed `<IIHIf` (session, length_after=keep+L, L, payload_bytes, box_seconds)          | h19 BF16 [L,4,5120] (40,960 B/row) ‖ pre F32 [L,4] (16 B/row) ‖ ckv20 U8 [L,288] (Mac slot-2 packing) ‖ idxk20 U8 [L,68] (Mac slot-3 packing) — concatenated row-major, in that order |
| CLOS | Mac->box | JSON {session}                                                         | - |
| TENS | box->Mac | (stream:1) JSON {name, dtype, shape, offset, nbytes}: a byte range of one state tensor, split-wire format | raw bytes |
| PROG | box->Mac | (stream:1) JSON {tokens_done} after every prefill chunk                 | - |
| MANI | box->Mac | (stream:1) the state manifest (same JSON as STAT's safetensors metadata) | - |
| END  | box->Mac | (stream:1) JSON {tensors, bytes, prefill_s}                            | - |
| ERR  | box->Mac | JSON {error, retry}  (retry=true: capacity or draining; reconnect and OPEN again) | - |

Semantics
- The session length after OPEN is N-1 (state of tokens[:-1], same boundary as phase 2). The first STEP normally has keep = N-1 and L = 1 + drafts (the final prompt token plus DSpark drafts), matching the Mac's kickoff step.
- STEP with keep < current length is the rollback: rows > keep are discarded on the box (KV/SWA/index/compressor tails/Engram history rewound), then the new rows are appended. keep == current length means all previous rows were accepted.
- Row i of the STPR payload corresponds to position keep+i. h19/pre are the hyper-connection stream and pre-mix entering layer 20 for that row (what the Mac feeds to its layer 20); ckv20/idxk20 are the rows the Mac appends to its layer-20 global KV / index K caches (packed exactly as the Mac's pack_activation, ratio 1, compressed rope at that position).
- L is 1..8 (graph-captured sizes 1..8; larger falls back to eager).
- The box keeps everything for layers 0-19 (SWA windows, compressed KV/index rows of layers 2/8/14 and their pair tails, Engram 3-token history) per session on the GPUs. Sessions: up to 2 concurrent at 512K each (budget), more at shorter contexts; a STEP on a session while another session steps is serialized (one GPU batch per step; batching two sessions into one forward is a later optimization).
- Errors on a session close it (ERR then socket close).

Streaming and prefix cache (sprint 4, box engine s4)
- `stream:1`: ACK comes first; the layer 2/8/14/20 KV and index rows (slots 2/3, ~99% of the bytes) go out as `TENS`
  parts as each 8K prefill chunk finishes, then the final tensors (windows, ratio-2 tails, tail.hidden/pre, the fixed
  slots), `MANI`, `END`. The tensors are byte-identical to the STAT file of the same prompt. The transfer overlaps the
  prefill; only ~12 MB (tail.hidden + windows) is left after the last chunk. Parser: split-wire's (the Mac's
  `pipe_wire.EncoderSession.open` already accepts it).
- `state:"lean"`: layers 0-19 carry only slot 0 (the offset) and empty slots (manifest compress_ratio 0, `state:"lean"`);
  layer 20 slots 0-3, tail.hidden/pre and tokens as in "full". ~0.43 KB/token instead of ~0.9.
- `delta_from:P` + `prefix_sha256` (sha256 of uint32 LE tokens[:P]; ERR on mismatch): the row tensors (slots 2/3 of the
  sent source layers) hold only rows [P // ratio, (N-1) // ratio); manifest `delta_from: P`. For a Mac that keeps
  rows [0, P) of that exact prefix.
- Every manifest and ACK carries `numerics` (now "og-s4.3"); key cached rows by it. og-s4.3 = exact top-k ties resolved
  lowest index first (the fused top-k's radix tie path used atomic arrival order: long prompts were nondeterministic)
  on top of og-s4.2 = the dsv41 RMSNorm on one reduction for every row count (steps changed in rare rows, states
  unchanged) on top of og-s4.1 = the 39c9e73 arithmetic
  with no 1-row prefill chunk (chunks follow the 8K grid; a 1-row remainder borrows a row from its neighbour), which
  changes the bytes of prompts with (N-1) % 8192 == 1 only (they now equal the prefix of a longer prompt).
- Snapshot points: every cache=1 prompt end (N-1) and every 8K grid point that prompt's prefill passes (grid points
  resume only with >= 256 new tokens). The native pages are stored once per 8K block and shared between entries.
  Files live in /dev/shm/split-nv/cache/<numerics>/ and survive an engine/container restart (not a reboot).
- Admin (HTTP 10051, only from 10.10.10.2 / localhost): `POST /admin/restart` (drain, exit; the unit restarts the
  engine, ~2.5 min), `POST /admin/crash` (exit 1 immediately; simulates a crash).
- `cache:1`: the box resumes from the longest host-RAM snapshot whose tokens are a prefix of `prompt[:-1]` (at least
  SPLIT_NV_CACHE_MIN_RESUME=1024 tokens), prefills the rest and snapshots the session at the end of this prompt
  (LRU, SPLIT_NV_CACHE_GB=64 for both TP ranks + the packed rows). The state bytes and all later STEP payloads are
  identical to a fresh prefill of the same prompt. ACK/manifest timing report `resumed_tokens`.
- Health: `GET http://10.10.10.1:10051/health` -> 200 {ok:true, encoder:"up", step_api:"up", sessions, cache, version, ...}
  or 503 while draining / when a GPU job is stuck (>60 s). `GET /v1/cache` = cache stats.
- Restart semantics: SIGTERM (systemctl --user stop/restart split-nv-engine) drains: new OPENs get `ERR {retry:true}`,
  open sessions get up to 30 s, then the engine exits. The user unit restarts it after a crash (15 s + ~2.5 min load)
  and starts it at boot. A GPU job stuck for 300 s makes the engine exit (watchdog) so it restarts.

Scheduling (s4): a STEP is served before the next prefill chunk of another session. While any other session has stepped
within the last second, prefill chunks are split into 2048-row pieces (SPLIT_NV_SHARE_CHUNK), so a decoding session
waits for at most one piece. Chunk geometry does not change any byte (M >= 2 invariance; no 1-row pieces).
Deploys run from the pinned worktree /home/ian/split-nv-deploy (tools/s4_deploy.sh <commit> <label>); the nested
sglang tree and model files come from /home/ian/split-nv.

Images (vision, box f1b5ebe)
- OPEN `images`: one entry per image span of the prompt. Every position of a span carries image_token_id (129264);
  length = ceil(vit_h/3) * (ceil(vit_w/3) + 1) + 2 ([IMAGE_START] + ([IMAGE] * nw + [NEWLINE]) * nh + [IMAGE_END]).
  The payload after the N token ids holds each image's patches: bf16 [vit_h * vit_w, 3, 14, 14] row-major (the
  official image_processor / the Mac Processor's pixel_values; byte-identical for the example images).
  sha256 = sha256(uint32 LE vit_h || vit_w || patch bytes). Spans lie inside prompt[:-1] and do not overlap; a prompt
  with image_token_id outside a declared span, or a digest mismatch, gets ERR {retry:false}.
- The box runs the DS-V4.1 vision tower + aligner (the SGLang fork's port of inference/vision.py, original BF16
  weights; relRMS 0.027 vs the official code, within its own SDPA-backend spread of 0.018-0.024) and replaces the
  embeddings of the span with its rows (learned image_start/newline/end at the delimiters). Image positions route
  with bias_vl and take no Engram contribution (inference/model.py); the Mac passes the same mask (ids ==
  image_token_id) to layers 20-39 in its tail replay. Text-only prompts are byte-identical to before.
- Cache keys are uint64: token ids, and (1<<63) | 63 bits of sha256(image digest || offset LE u32) inside image spans
  (hooks/split_nv/imagekeys.py; the Mac's og_images.py is the same, test vector in tools/test_imagekeys.py). With
  images, `prefix_sha256` of a `delta_from` OPEN is sha256 of the uint64 LE keys[:P]. A resume never crosses an image
  whose content differs; turn 2 after an image skips the ViT.
- The manifest echoes `images` [{start, length, grid, sha256}].
