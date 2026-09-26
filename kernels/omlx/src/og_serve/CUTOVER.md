# ds41-og cutover plan (proposal; not applied)

Goal: clients that ask for `ds41` (Hermes Agent and aliases) get the original-weight split
pipeline (og); the Mac-only q3 build stays available as `ds41-q3`. One config edit, reversible by
restoring one file. The owner decides and applies it.

## Preconditions

1. The coordinator has merged og-serve (and og-cache / og-speed if ready) into `~/src/wt/ds41-og`
   and rsynced the git-ignored kernels. `~/llm/ds41/og/ds41-og` execs
   `$DS41_TREE/og_serve/ds41_og.py`, so it picks up the merged supervisor without edits. Check:
   `diff ~/llm/ds41/og/ds41-og ~/src/wt/ds41-og/og_serve/ds41-og` shows no difference.
2. Box: og-box's supervised engine is up (`nc -z 10.10.10.1 10052`), with auto-restart after a
   crash and after a reboot.
3. Gates are recorded in PROGRESS.md: parity suite, soak, benchmark, quality. Measured by og-serve
   on ds41-og 45ef51f8 / 3e1ea5a8 + box d11dccf (og-s4.3):
   - parity 19/19 (q3 20/20);
   - TTFT 8K 0.96 s vs q3 3.35 s; 128K 7.81 s vs 51.6 s;
   - resume turn 2 through the API: 8K 0.54 s, 128K 0.94 s (q3 1.0 s). The exact repeat is
     identical, at 0.43 / 0.88 s;
   - c1 decode 8K 90.7 vs 70.5 tok/s; 128K 80.7 vs 70.8 tok/s;
   - c2 total 8K 127.6 vs 86.6 tok/s;
   - 5-min soak: 128 requests, 0 errors, footprint 149.3-150.6 GiB;
   - Hermes Agent: correct;
   - split1 quality: NLL 0.0507 vs 0.0623, needles 128K and 1M recalled.

**Blockers: none open.**
- Fixed: the exact repeat of a turn of 4K tokens or more used to fail (og-cache 3e1ea5a8). Verified
  in sv-V1: repeats at 8K and 128K are identical, with 0 errors.
- Merge before the cutover: og-serve 080c5a38. If a box-state import fails, it retries once with a
  plain full OPEN and never falls into a local prefill.
- Not tested (owner cut): a box reboot. The engine unit is enabled at boot, and the supervisor
  serves q3 until the box answers.

## Change: `~/llm/llama-swap.yaml`

Exact target file: `og_serve/llama-swap.cutover.yaml`. Diff: `og_serve/llama-swap.cutover.diff`.
Both were regenerated from the live file at 20:37 UTC, which already had the "Icculus" rename. If
llama-swap.yaml changes again, re-apply the same edits instead of copying the file.

- The og entry's key goes from `ds41-og` to `ds41`. It keeps the same `cmd` (CPU supervisor, no
  `${gpu}` prefix, because each child takes gpu.lock through gpu-exec), `env DS41_TREE`,
  `useModelName: ds41-og` and `ttl: 0`, and the owner's display name "Icculus · ...". Its aliases
  become `[icculus, deepseek-v4.1-flash, ds41-og, deepseek-v4.1-flash-og]`, and it keeps the
  `${MODEL_ID}:think` filter.
- The q3 entry's key goes from `ds41` to `ds41-q3`. `cmd` and `env` are unchanged. Its aliases
  become `[ds41-q3g128, ds41-c2, deepseek-v4.1-flash-c2, deepseek-v4.1-flash-q3]`, and it keeps the
  `:think` filter.
- `hooks.on_startup.preload: [ds41]` is unchanged, so it now preloads og. When the box is down at
  startup, the supervisor loads q3 instead and switches to og once the box has been up for
  DS41_OG_RECOVER_S (60 s) and no request is in flight.
- The header comment lines are updated. `ds41-c1` and every other entry are unchanged.

Apply:
```sh
cp ~/llm/llama-swap.yaml ~/llm/llama-swap.yaml.pre-og-cutover
cp ~/src/wt/ds41-og/og_serve/llama-swap.cutover.yaml ~/llm/llama-swap.yaml   # llama-swap runs with -watch-config
```
llama-swap reloads the file itself. The reload stops the running q3 `ds41` process and the preload
starts the og supervisor: about 25 s until ready, plus up to 45 s if q3 still holds the lock while
it drains.

Verify (about 2 min):
```sh
curl -s localhost:8080/running            # ds41 ready, cmd .../ds41/og/ds41-og
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"ds41","messages":[{"role":"user","content":"Say ready."}],"max_tokens":8}' -D - | grep -i x-ds41-og-backend   # og
~/llm/.venv-ds41-omlx-tiles/bin/python ~/src/wt/ds41-og/og_serve/parity.py --label cutover-check \
  --target og=http://127.0.0.1:8080/v1@ds41 --think-model og=ds41:think
```
The parity run should report `failures: {"og": []}`. Its `think` case also exercises `ds41:think`
through llama-swap's filter.

## Hermes Agent

No change is needed. `~/.hermes/config.yaml` has `model.default: "ds41"` and base_url
`http://127.0.0.1:8080/v1`, so it gets og after the reload. Its stream-stale limit for local
providers is 900 s and its request timeout is 1800 s. Both are well above the worst failover pause.
An engine restart pauses a stream for up to about 2.5 min. A lost box host means a 45 s tier-A
wait, then 43 s for the q3 load plus q3's prefill of the conversation (8K 3 s, 128K 51 s). The
stream keeps receiving SSE keepalive chunks throughout. To pin Hermes to q3 without a
rollback, set `model.default: "ds41-q3"`.

## Rollback (under 1 min, no data loss)

```sh
cp ~/llm/llama-swap.yaml.pre-og-cutover ~/llm/llama-swap.yaml    # watch-config reloads; ds41 = q3 again
```
The reload stops the og supervisor, which stops its child by PID and waits for it, and the preload
loads q3 (43 s). Hermes needs no change. If only the box is the problem, nothing needs rolling back:
the supervisor already serves q3 while the box is down and returns to og afterwards.

## Behaviour after the cutover (what the owner should expect)

- **Box engine restarting.** The host answers but the port is refused, as during a systemd restart
  after a crash or a deploy: 22-52 s of drain plus about 2 min of load. og keeps serving.
  In-flight streams pause, then continue bit-identically on the rebuilt box sessions. New requests
  wait in the og worker for up to DS41_OG_RESTART_WAIT_S (240 s). There are no model swaps.
- **Box draining or full.** The OPEN gets a retryable ERR and is retried for up to
  DS41_OG_BUSY_WAIT_S (300 s).
- **Box host or link down** (connect timeout or unreachable). A new request goes to q3: the
  supervisor probes the box before each request, waits for og's in-flight requests, stops og and
  loads q3 (43 s). An in-flight stream waits DS41_OG_RESUME_WAIT_S (45 s) in tier A before tier B.
- **Tier A: box fails mid-stream but comes back.** The og worker rebuilds the box session from the
  tokens already committed and resends the step. It uses the box prefix cache when the box has
  one. The client sees only a pause: about 1 s for a link drop (measured). The output is
  bit-identical to an uninterrupted run (measured at 8K for a link kill and for an 8 s outage).
- **Tier B: box stays down.** Each affected request ends inside the og worker with a resume marker
  that carries its generated tokens. The supervisor keeps the client stream open, sending
  keepalives. It loads q3 and resends the request with those tokens; q3 replays them through its
  normal output path, and the client receives only the text it has not seen yet, under the same
  response id. `usage.prompt_tokens` excludes the replayed tokens. This happens at most
  DS41_OG_MAX_RESUMES (2) times per request, within DS41_OG_FAILOVER_S (900 s). Measured c2 at 8K:
  both streams finished with no error and 0 divergent characters.
- **Neither works**: the stream ends with a clean, retryable error. It is an SSE event
  `{"error": {"code": "backend_unavailable", ...}}` followed by `[DONE]`. A non-streaming request
  gets HTTP 503 with `Retry-After: 30`. A child that dies mid-response (no marker) ends the same way.
- **og cannot start** (for example the box drops again during og's warm-up): the supervisor serves
  from q3.
- **Box back**: q3 hands back to og once the box has answered probes for 60 s and nothing is in
  flight. The switch takes about 26 s: 2 s to stop q3 and 24 s to load og.
- **Failed box OPEN**: only that request fails (it is resumed or replayed on q3). Other streams
  continue.
- **Memory**: only one child is resident at a time, og at 150 GiB or q3 at 224 GiB. The og worker
  has its own 245 GiB SIGKILL watchdog. The supervisor attaches the same watchdog to q3 once q3 is
  ready. Each child also has an in-process 245 GiB guard.
- **Observability**: the supervisor's `GET /health` (through llama-swap: `/upstream/ds41/health`)
  shows the backend, the child pid, box_up/box_trusted and the counters failovers, box_lost,
  resumed, resume_failed, broken and divergent. The og worker's `/og/stats` shows box sessions,
  recoveries and resume counters. Logs are in `~/llm/ds41/og/logs/{og,q3}-child.log`.

## Known limits

- Resume covers `/v1/chat/completions` and `/v1/completions`. A box loss during `/v1/messages` or
  `/v1/responses` ends that response with an error (Hermes uses chat completions). Requests with
  `logprobs` or `n > 1` are not resumed.
- Tier A is bit-identical as long as the box's step and prefill arithmetic agree. On boxes
  before s4, a 1-row prefill chunk rounded differently, which affected a rebuild at
  `keep % 8192 == 1`. s4 (numerics og-s4.1) no longer uses 1-row chunks.
- While a long prompt (for example 512K) prefills on the box, the other session's decode steps
  wait behind each 8K prefill chunk. That session drops to about 6-20 tok/s until the prefill
  ends: a box scheduling limit, not an error.
- After a failover to q3, the rest of the response comes from the q3 weights, so it is
  numerically different. A long context on q3 pays q3's prefill: 512K takes about 4 min, still
  inside the 900 s failover budget and Hermes' 900 s stale limit.
