"""mHC kernels: latency of hc_fuse.project_pre_norm / post_mix variants (small: hc weights of 3 layers)."""
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MLX_ENABLE_TF32', '0')
os.environ.setdefault('DS41_MHC', '1')
import mlx.core as mx

from omlx.patches.deepseek_v41 import ffn_fuse, hc_fuse
from omlx.patches.deepseek_v41.loading import _load_shard

HOME = Path.home()
MODEL = HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'
OUT = open(os.environ.get('HB_OUT', str(HOME/'llm/ds41/ffn/hc.jsonl')), 'a')


def emit(**r):
    line = json.dumps(r); OUT.write(line + '\n'); OUT.flush(); print(line, flush=True)


def variant_kernel(u):
    src = hc_fuse._PROJECT_PRE_NORM.replace('constexpr uint U = 8;', f'constexpr uint U = {u};')
    return mx.fast.metal_kernel(name=f'hcpp_u{u}', input_names=['x', 'pre', 'fn', 'weight', 'eps'],
                                output_names=['mix', 'y'], header=hc_fuse._HEADER, source=src)


def split_kernel(j, quant):
    """pre-norm spread over J threadgroups per row (each recomputes the norm, writes 1/J of the row)."""
    from omlx.patches.deepseek_v41 import ffn_fuse as F
    body = F._PRE_NORM_Q if quant else F._PRE_NORM_Q.replace(
        "device float* xf, device T* xq)", "device T* y)").replace(
        "        xf[row * D + d] = v;\n        xq[row * D + d] = T(ds41_fp8_round(v));\n", "        y[row * D + d] = T(v);\n")
    body = body.replace("inline void ds41_pre_norm_row_q(", "inline void ds41_pre_norm_row_j(uint js, ").replace(
        "template <typename T, uint D, typename PX, typename PP, typename PW>\ninline void ds41_pre_norm_row_j",
        "template <typename T, uint D, uint J, typename PX, typename PP, typename PW>\ninline void ds41_pre_norm_row_j").replace(
        "    for (uint t = 0; t < (D + 255) / 256; ++t) {\n        const uint d = tid + 256 * t;\n        const float v",
        "    for (uint t = js * ((D + 255) / 256) / J; t < (js + 1) * ((D + 255) / 256) / J; ++t) {\n        const uint d = tid + 256 * t;\n        const float v")
    assert "ds41_pre_norm_row_j" in body and "js * (" in body, body
    call = ("ds41_pre_norm_row_j<T, D, J>(o - 24, x, pre, weight, eps[1], row, t, lane, sg, sums, xf, xq);" if quant
            else "ds41_pre_norm_row_j<T, D, J>(o - 24, x, pre, weight, eps[1], row, t, lane, sg, sums, y);")
    src = hc_fuse._PROJECT_PRE_NORM.replace("if (o == 24) {", "if (o >= 24) {").replace(
        "ds41_pre_norm_row<T, D>(x, pre, weight, eps[1], row, t, lane, sg, sums, y);", call)
    return mx.fast.metal_kernel(name=f'hcpp_j{j}_q{int(quant)}', input_names=['x', 'pre', 'fn', 'weight', 'eps'],
                                output_names=['mix', 'xf', 'xq'] if quant else ['mix', 'y'],
                                header=hc_fuse._HEADER + F._ROUND + body, source=src)


def main():
    wm = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    layers = []
    for i in (20, 21, 22):
        pre = f'language_model.layers.{i}.'
        d = {}
        for name in ('hc_ffn_fn', 'hc_ffn_base', 'hc_ffn_scale', 'ffn_norm.weight'):
            d[name] = _load_shard(MODEL/wm[pre + name])[pre + name]
        layers.append(d)
    mx.eval(layers)
    snap = mx.load(str(HOME/'llm/ds41/og-speed/snap-8k/step.safetensors'))
    H, P = snap['h'], snap['pre']
    kernels = {u: variant_kernel(u) for u in (8, 16)}
    splits = {(j, q): split_kernel(j, q) for j in (1, 2, 4, 5, 10, 20) for q in (False, True)}
    eps = hc_fuse._consts(1e-6, 1e-20)
    calls = 60
    for rows in (1, 5):
        h0, p0 = H[:, :rows], P[:, :rows]
        # bitwise: variants vs production
        ref = hc_fuse.project_pre_norm(h0, p0, layers[0]['hc_ffn_fn'], layers[0]['ffn_norm.weight'], 1e-6, 1e-20)
        for u, k in kernels.items():
            out = k(inputs=[h0, p0, layers[0]['hc_ffn_fn'], layers[0]['ffn_norm.weight'], eps],
                    template=[('T', h0.dtype), ('D', 5120), ('DH', 20480)], grid=(rows * 256, 25, 1),
                    threadgroup=(256, 1, 1), output_shapes=[(1, rows, 24), (1, rows, 5120)],
                    output_dtypes=[mx.float32, mx.bfloat16])
            mx.eval(out, ref)
            emit(test='bitwise', rows=rows, u=u, ok=all(bool(mx.array_equal(a, b).item()) for a, b in zip(out, ref)))

        refq = ffn_fuse.pre_norm_q(h0, p0, layers[0]['hc_ffn_fn'], layers[0]['ffn_norm.weight'], 1e-6, 1e-20)
        for (j, q), k in splits.items():
            out = k(inputs=[h0, p0, layers[0]['hc_ffn_fn'], layers[0]['ffn_norm.weight'], eps],
                    template=[('T', h0.dtype), ('D', 5120), ('DH', 20480), ('J', j)], grid=(rows * 256, 24 + j, 1),
                    threadgroup=(256, 1, 1),
                    output_shapes=[(1, rows, 24), (1, rows, 5120)] + ([(1, rows, 5120)] if q else []),
                    output_dtypes=[mx.float32] + ([mx.float32, mx.bfloat16] if q else [mx.bfloat16]))
            r = refq if q else ref
            mx.eval(out, r)
            emit(test='bitwise_split', rows=rows, j=j, quant=q, ok=all(bool(mx.array_equal(a, b).item()) for a, b in zip(out, r)))
        tiny = mx.array(1e-30, mx.bfloat16)
        routed = mx.zeros((rows * 6, 5120), mx.bfloat16)
        shared = mx.zeros((rows, 5120), mx.bfloat16)
        mixes = [hc_fuse.project_pre_norm(h0, p0, L['hc_ffn_fn'], L['ffn_norm.weight'], 1e-6, 1e-20)[0] for L in layers]
        mx.eval(routed, shared, mixes)

        def chain(kind, u=8):
            def run():
                h, p = h0, p0
                acc = None
                for c in range(calls):
                    L = layers[c % 3]
                    if kind == 'pre_norm':
                        hh = h if acc is None else h + acc
                        mix, y = kernels[u](inputs=[hh, p, L['hc_ffn_fn'], L['ffn_norm.weight'], eps],
                                            template=[('T', h.dtype), ('D', 5120), ('DH', 20480)],
                                            grid=(rows * 256, 25, 1), threadgroup=(256, 1, 1),
                                            output_shapes=[(1, rows, 24), (1, rows, 5120)],
                                            output_dtypes=[mx.float32, mx.bfloat16])
                        acc = (y[:, :, None, :1] * tiny)
                    elif kind.startswith('split'):
                        j, q = u
                        hh = h if acc is None else h + acc
                        out = splits[u](inputs=[hh, p, L['hc_ffn_fn'], L['ffn_norm.weight'], eps],
                                        template=[('T', h.dtype), ('D', 5120), ('DH', 20480), ('J', j)],
                                        grid=(rows * 256, 24 + j, 1), threadgroup=(256, 1, 1),
                                        output_shapes=[(1, rows, 24), (1, rows, 5120)] + ([(1, rows, 5120)] if q else []),
                                        output_dtypes=[mx.float32] + ([mx.float32, mx.bfloat16] if q else [mx.bfloat16]))
                        acc = (out[-1][:, :, None, :1] * tiny)
                    elif kind == 'pre_norm_q':
                        hh = h if acc is None else h + acc
                        mix, xf, xq = ffn_fuse.pre_norm_q(hh, p, L['hc_ffn_fn'], L['ffn_norm.weight'], 1e-6, 1e-20)
                        acc = (xq[:, :, None, :1] * tiny)
                    elif kind == 'post':
                        h, p = ffn_fuse.post_combine(routed, shared, h, mixes[c % 3], L['hc_ffn_scale'],
                                                     L['hc_ffn_base'], 1e-6, 20, 6)
                    elif kind == 'glue':
                        hh = h if acc is None else h + acc
                        acc = hh[:, :, :, :1] * tiny
                return (h, p) if acc is None else acc
            return run
        specs = [('glue', 8), ('post', 8), ('pre_norm_q', 8)] + [('pre_norm', u) for u in kernels] + [('split', ju) for ju in splits]
        t = {s: [] for s in specs}
        for rep in range(12):
            for s in (specs if rep % 2 == 0 else specs[::-1]):
                fn = chain(*s)
                mx.synchronize()
                s0 = time.perf_counter()
                mx.eval(fn())
                if rep >= 2:
                    t[s].append((time.perf_counter() - s0) * 1000)
        emit(test='hc_chain', rows=rows, per_call_us={f'{k}_u{u}': round(statistics.median(v) / calls * 1000, 1)
                                                      for (k, u), v in t.items()})


main()
