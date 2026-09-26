"""Request-local greedy DSpark loop with transactional encoder rollback.

Each generator resumption receives the box rows of the step sent earlier, runs
layers 20-39 + head, verifies, drafts and sends the next STEP before yielding.
A scheduler that round-robins two requests therefore overlaps the box forward
of one request with the Mac forward of the other.
"""
import os
import time
import mlx.core as mx

from .mtp import AcceptanceDepthController
from .pipe_wire import EncoderSession, mlx_state, mlx_step
from ..mlx_lm_mtp.deepseek_v4_dspark import capture_prompt
from ..mlx_lm_mtp.copy_draft import CopyIndex

# Pipeline verify cycle C(L), L=2..5 rows, in ms: box step + link + Mac layers
# 20-39/head + drafter. Same (context tier -> costs) shape as the served table.
# Re-profiled for og-speed (Sep 25): box 1f592ea step RTT from the Mac
# ~9.9/10.5/10.6/11.1 + Mac layers 20-39/head 14.7/17.2/18.9/20.7 (one-pass
# head, MXFP4 MoE launches) + width-4 drafter 4.3 + accept 0.3; long contexts
# add ~0.3/0.6-1.1 ms of Mac attention. The flatter curve favours deeper drafts.
_PIPE_COSTS = ((524288, (30.2, 33.4, 35.4, 37.8)),
               (131072, (29.9, 32.9, 34.7, 37.1)),
               (0, (29.6, 32.6, 34.4, 36.7)))


# Calibration log (JSONL path): per DSpark cycle the drafter's confidences,
# the verified depth and how many drafts the target accepted. Off by default.
_CALIB = os.environ.get('DS41_PIPE_CALIB', '')


def _pipeline_costs():
    raw = os.environ.get('DS41_PIPE_COSTS', '')
    if not raw:
        return _PIPE_COSTS
    tiers = []
    for part in raw.split(';'):
        floor, values = part.split(':')
        costs = tuple(float(x) for x in values.split(','))
        if len(costs) != 4 or any(c <= 0 for c in costs):
            raise ValueError('DS41_PIPE_COSTS needs four positive L=2..5 costs per tier')
        tiers.append((int(floor), costs))
    return tuple(sorted(tiers, reverse=True))


class PipelineDepthController(AcceptanceDepthController):
    """Served EVICT rule with the pipeline's own profiled C(L)."""
    costs = _pipeline_costs()

    def choose_cost_depth(self, probabilities, context):
        if not self.cost_policy or context < 1024:
            return self.cur
        costs = next(c for floor, c in self.costs if context >= floor)
        cumulative, expected, best, best_utility = 1.0, 1.0, 1, -1.0
        for index, probability in enumerate(probabilities[: self.max_depth]):
            cumulative *= max(0.0, min(1.0, float(probability)))
            expected += cumulative
            utility = expected / costs[index]
            if utility > best_utility:
                best, best_utility = index + 1, utility
        return best


def open_remote(encoder, tokens, request_id='', **options):
    """Network half of admission (box prefill + state transfer); no MLX calls."""
    start = time.perf_counter()
    tensors, manifest = encoder.open(tokens, request_id, **options)
    return tensors, manifest, time.perf_counter() - start


class PipelineRequest:
    def __init__(self, model, tokens, *, encoder=None, request_id='', opened=None):
        self.model, self.tokens = model, list(tokens)
        self.encoder = encoder if encoder is not None else EncoderSession()
        self.cache = self.mtp_cache = None
        self.steps = []
        self.accepted = self.proposed = self.copy_accepted = self.copy_proposed = 0
        self.controller = PipelineDepthController(4)
        self.copy = None
        self._pending = None
        try:
            if opened is None:
                opened = open_remote(self.encoder, tokens, request_id)
            tensors, manifest, self.prefill_remote_s = opened
            start = time.perf_counter()
            arrays = mlx_state(tensors)
            self.cache = model.import_prefill(arrays,manifest,tokens,identity=self.encoder.identity)
            mx.eval([x for item in self.cache for x in item.cache if x is not None])
            self.replay_s = time.perf_counter()-start
        except BaseException:
            self.encoder.close()
            raise

    def close(self):
        self.encoder.close()
        self.cache = self.mtp_cache = None

    def submit(self, ids):
        start = self.cache[0].size()
        self.encoder.send_step(ids, start)
        self._pending = (list(ids), start)

    def complete(self, *, verify=False, greedy=False):
        ids, start = self._pending
        self._pending = None
        raw, timing = self.encoder.recv_step()
        t = time.perf_counter()
        arrays = mlx_step(raw,len(ids))
        logits, hidden = self.model.forward_boundary(**arrays,cache=self.cache,start=start,verify=verify)
        if greedy:
            # Greedy targets in the same command buffer as the forward.
            logits = mx.argmax(logits[0],-1)
        mx.eval(logits,hidden)
        timing['mac_s'] = time.perf_counter()-t
        self.steps.append(timing)
        return logits,hidden

    def forward(self, ids, *, verify=False):
        self.submit(ids)
        return self.complete(verify=verify)

    def draft(self, hidden, committed, remaining):
        start=time.perf_counter()
        try:
            return self._draft(hidden,committed,remaining)
        finally:
            self.steps[-1]['draft_s']=time.perf_counter()-start

    def _draft(self, hidden, committed, remaining):
        self.copy.append(committed)
        copies = self.copy.propose(remaining-1)
        if copies:
            self.model.dspark_append_context(hidden,self.mtp_cache)
            mx.eval([x.keys for x in self.mtp_cache])
            return copies[:4], 'copy'
        cost = self.controller.cost_policy and self.mtp_cache[0].offset >= 1024
        width = self.controller.max_depth if cost else self.controller.cur
        logits,_ = self.model.dspark_forward(hidden,mx.array([[committed[-1]]],mx.uint32),
                                             self.mtp_cache,draft_length=width)
        previous = mx.array([committed[-1]],mx.uint32)
        tokens, probs = [], []
        for i in range(width):
            bias,_ = self.model.dspark_markov(previous)
            scores = logits[:,i]+bias
            previous = mx.argmax(scores,-1).astype(mx.uint32)
            tokens.append(previous)
            probs.append(mx.exp(mx.max(scores)-mx.logsumexp(scores)))
        chosen = mx.concatenate(tokens)
        mx.eval(chosen,probs)
        probs = [float(p.item()) for p in probs]
        depth = self.controller.choose_cost_depth(probs,self.mtp_cache[0].offset) if cost else width
        self._draft_probs = (probs, self.mtp_cache[0].offset)
        return chosen[:depth].tolist(), 'dspark'

    def generate(self,max_tokens,*,ignore_eos=False):
        """Yield committed token batches, each with the next STEP already sent.

        The first resumption only sends the kickoff step and yields []. The
        anchor remains outside both caches.
        """
        try:
            self.submit([self.tokens[-1]])
            yield []
            logits,hidden = self.complete()
            capture_prompt(self.model,mx.array([[self.tokens[-1]]],mx.uint32),hidden,self.cache)
            first = int(mx.argmax(logits[0,-1]).item())
            if max_tokens <= 1 or (first == 1 and not ignore_eos):
                yield [first]
                return
            self.submit([first])
            yield [first]
            logits,hidden = self.complete()
            anchor = int(mx.argmax(logits[0,-1]).item())
            primed = self.model.mtp_take_primed(self.cache,first)
            if primed is None:
                raise ValueError('DSpark prompt context is not aligned')
            self.mtp_cache,_ = primed
            self.copy = CopyIndex(self.tokens+[first])
            emitted = 2
            if emitted >= max_tokens or (anchor == 1 and not ignore_eos):
                yield [anchor]
                return
            drafts,source = self.draft(hidden,[anchor],max_tokens-emitted)
            self.submit([anchor,*drafts])
            yield [anchor]
            while True:
                targets,hidden = self.complete(verify=True,greedy=True)
                targets = targets.tolist()
                accepted = 0
                while accepted < len(drafts) and drafts[accepted] == targets[accepted]:
                    accepted += 1
                committed = drafts[:accepted]+[targets[accepted]]
                committed = committed[:max_tokens-emitted]
                if not ignore_eos and 1 in committed:
                    committed = committed[:committed.index(1)+1]
                accepted = len(committed)-1
                self.model.rollback_boundary(self.cache,accepted+1)
                self.steps[-1]['accepted'] = accepted
                self.steps[-1]['source'] = source
                if _CALIB and source == 'dspark':
                    import json
                    probs, context = self._draft_probs
                    with open(_CALIB, 'a') as f:
                        f.write(json.dumps(dict(probs=probs, context=context, used=len(drafts),
                                                accepted=accepted, full=len(committed) == len(drafts) + 1)) + '\n')
                if source == 'copy':
                    self.copy_proposed += len(drafts)
                    self.copy_accepted += accepted
                    self.copy.observe(accepted)
                else:
                    self.proposed += len(drafts)
                    self.accepted += accepted
                emitted += len(committed)
                if emitted >= max_tokens or (committed[-1] == 1 and not ignore_eos):
                    yield committed
                    return
                old_k = len(drafts)
                drafts,source = self.draft(hidden[:,:accepted+1],committed,max_tokens-emitted)
                self.controller.observe(old_k,accepted,0)
                anchor = committed[-1]
                self.submit([anchor,*drafts])
                yield committed
        finally:
            self.close()

    def summary(self):
        return dict(prefill_remote_s=self.prefill_remote_s,replay_s=self.replay_s,
                    open_info=self.encoder.open_info,
                    steps=self.steps,dspark_accepted=self.accepted,dspark_proposed=self.proposed,
                    copy_accepted=self.copy_accepted,copy_proposed=self.copy_proposed)

    def teacher_force(self,target_ids,*,block=4):
        if block != 4:
            raise ValueError('Quality baseline uses fixed L4')
        inputs = [self.tokens[-1]]+list(target_ids[:-1])
        rows=[]
        try:
            for a in range(0,len(inputs),block):
                ids=inputs[a:a+block]
                logits,_ = self.forward(ids+[1]*(block-len(ids)),verify=True)
                rows.append(logits[0,:len(ids)].astype(mx.float32))
            result=mx.concatenate(rows)
            mx.eval(result)
            return result
        finally:
            self.close()
