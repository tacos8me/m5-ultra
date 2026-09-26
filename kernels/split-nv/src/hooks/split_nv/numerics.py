"""Bounded eager layer trace to localize prefill/step mismatches."""
import json
import torch


def metrics(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return dict(rel_rms=((a-b).norm()/b.norm().clamp_min(1e-12)).item(),
                max_abs=(a-b).abs().max().item())


def record(cap, key, h):
    if cap.trace is None:
        return
    p = cap.trace_positions[cap.trace_mask].clone()
    h = h[cap.trace_mask].clone()
    if key in cap.trace:
        p0,h0 = cap.trace[key]
        p,h = torch.cat((p0,p)),torch.cat((h0,h))
    cap.trace[key] = p,h


def configure(engine, variant):
    cap = engine.cap
    if variant == 'trace':
        from sglang.srt.layers.attention.dsv4 import dsv41_sparse as S
        orig_project = S.DeepseekV41Compressor.project
        def project(self,x):
            kv,score = orig_project(self,x)
            if cap.enabled and cap.trace is not None:
                record(cap, f'{self._pipe_layer}.compress.kv', kv)
                if score is not None:
                    record(cap, f'{self._pipe_layer}.compress.score', score)
            return kv,score
        S.DeepseekV41Compressor.project = project
        orig_weights = S.DeepseekV41Indexer.head_weights_raw
        def weights(self,x):
            out = orig_weights(self,x)
            if cap.enabled and cap.trace is not None:
                record(cap,f'{self._pipe_layer}.index.weights',out)
            return out
        S.DeepseekV41Indexer.head_weights_raw = weights
        for i, layer in enumerate(engine.mr.model.model.layers):
            if layer.self_attn.compressor is not None:
                layer.self_attn.compressor._pipe_layer = i
            if layer.self_attn.indexer is not None:
                layer.self_attn.indexer._pipe_layer = i
            modules = [(name,getattr(layer,name)) for name in ('input_layernorm','self_attn','post_attention_layernorm','mlp')]
            modules += [('mlp.'+name,getattr(layer.mlp,name,None)) for name in ('gate','experts','shared_experts')]
            modules += [('self_attn.'+name,getattr(layer.self_attn,name,None)) for name in ('wqkv_a','q_norm','wq_b','wo_b')]
            for name,module in modules:
                if module is None:
                    continue
                def hook(mod, args, output, key=f'{i}.{name}'):
                    if cap.enabled and cap.trace is not None:
                        record(cap, key, output[0] if isinstance(output,tuple) else output)
                module.register_forward_hook(hook)
    elif variant == 'sorted_topk':
        from sglang.srt.layers.attention import deepseek_v4_backend as B
        original = B.topk_transform_paged_v2
        def sorted_topk(scores,lens,pages,out,page_size,metadata):
            raw = torch.empty_like(out)
            original(scores,lens,None,raw,page_size,metadata)
            sentinel = torch.iinfo(raw.dtype).max
            raw = raw.masked_fill(raw < 0, sentinel).sort(-1).values
            valid = raw != sentinel
            safe = raw.masked_fill(~valid,0).long()
            slots = safe if pages is None else pages.gather(1,safe//page_size)*page_size + safe%page_size
            out.copy_(torch.where(valid,slots,-1))
        B.topk_transform_paged_v2 = sorted_topk
    elif variant == 'router_cublas':
        for layer in engine.mr.model.model.layers:
            layer.mlp.gate.tiny_router_gemm_max_tokens = 0
    elif variant == 'router_fixed':
        from sglang.srt.models.deepseek_v2 import MoEGate
        from split_nv.fixed_linear import linear
        def fixed(self,hidden_states,*args,**kwargs):
            return linear(hidden_states,self.weight)
        MoEGate.forward = fixed
    elif variant == 'compressor_fixed':
        from sglang.srt.layers.attention.dsv4 import dsv41_sparse as S
        from split_nv.fixed_linear import linear
        S.linear_bf16_fp32 = linear
    elif variant == 'indexer_fixed':
        from sglang.srt.layers.attention.dsv4 import dsv41_sparse as S
        from split_nv.fixed_linear import linear
        def weights(self,x):
            out = linear(x,self.weights_proj.weight).to(x.dtype)
            if cap.enabled and cap.trace is not None:
                record(cap,f'{self._pipe_layer}.index.weights',out)
            return out
        def wk(self,x):
            return linear(x,self.wk.weight).to(x.dtype)
        S.DeepseekV41Indexer.head_weights_raw = weights
        S.DeepseekV41Indexer.forward_wk = wk
    elif variant == 'attention_prefill':
        from sglang.kernels.ops.attention import flash_mla_sm120 as A
        original = A.flash_mla_with_kvcache_sm120
        def padded(*args, **kwargs):
            q = kwargs['q']
            n = q.shape[0]
            if n > 64:
                return original(*args, **kwargs)
            for name in ('q', 'indices', 'topk_length', 'extra_indices_in_kvcache', 'extra_topk_length'):
                x = kwargs.get(name)
                if x is not None:
                    value = -1 if 'indices' in name else 0
                    kwargs[name] = torch.cat((x, x.new_full((65-n, *x.shape[1:]), value)))
            out, aux = original(*args, **kwargs)
            return out[:n], aux
        A.flash_mla_with_kvcache_sm120 = padded
    else:
        raise ValueError(variant)


def snapshot(cap):
    return {k:(p.cpu(), h.cpu()) for k,(p,h) in cap.trace.items()}


def report(label, N, trace, ref):
    for k,(p,h) in trace.items():
        assert torch.equal(p,ref[k][0]), (label,k,p,ref[k][0])
        print(json.dumps(dict(diag=label, N=N, layer=k, **metrics(h,ref[k][1]))),flush=True)


def diagnose(front):
    cap = front.engine.cap
    tokens = json.load(open('/home/ian/split-nv/ref/ids-8192.json'))
    front.submit(('diagnostic','trace'))
    front.submit(('diagnostic','sorted_topk'))
    front.submit(('diagnostic','attention_prefill'))
    front.submit(('diagnostic','router_fixed'))
    for variant in ('compressor_fixed', 'indexer_fixed'):
        front.submit(('diagnostic',variant))
        for N in (256, 8190, 8192):
            diagnose_case(front, tokens, N, variant)


def diagnose_case(front, tokens, N, variant):
        cap = front.engine.cap
        L = 5
        cap.trace_start = N
        cap.trace = {}
        front.submit(('prefill', 0, tokens[:N+L], False))
        ref = snapshot(cap)
        front.submit(('close', 0))
        cap.trace = {}
        front.submit(('prefill', 0, tokens[:N+L], False))
        report(variant+'-prefill-repeat',N,snapshot(cap),ref)
        front.submit(('close', 0))
        cap.trace = None
        front.submit(('prefill', 0, tokens[:N], False))
        cap.trace = {}
        front.submit(('step', 0, N, tokens[N:N+L], False))
        first = snapshot(cap)
        report(variant+'-eager-prefill',N,first,ref)
        cap.trace = {}
        front.submit(('step', 0, N, tokens[N:N+L], False))
        report(variant+'-eager-reject-all',N,snapshot(cap),first)
        cap.trace = None
        front.submit(('step', 0, N, tokens[N:N+2]+[11,22,33], False))
        cap.trace_start = N+2
        cap.trace = {}
        front.submit(('step', 0, N+2, tokens[N+2:N+6], False))
        rejected = snapshot(cap)
        cap.trace = None
        front.submit(('close',0))
        front.submit(('prefill',0,tokens[:N],False))
        front.submit(('step',0,N,tokens[N:N+L],False))
        cap.trace = {}
        front.submit(('step',0,N+2,tokens[N+2:N+6],False))
        report(variant+'-partial-reject-vs-clean',N,rejected,snapshot(cap))
        cap.trace = None
        cap.trace_start = N
        front.submit(('close', 0))
        if N % 256 == 0:
            front.submit(('prefill', 0, tokens[:N], False))
            cap.trace = {}
            front.submit(('extend',0,tokens[N:N+L]))
            report(variant+'-extend-vs-verify',N,snapshot(cap),first)
            cap.trace = None
            front.submit(('close',0))
