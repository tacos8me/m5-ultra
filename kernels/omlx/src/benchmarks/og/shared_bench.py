"""Shared-expert gate/up (+SwiGLU/FP8) layouts: bitwise + chain timing (shared weights of 3 layers, ~110 MB)."""
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MLX_ENABLE_TF32', '0')
os.environ.setdefault('DS41_MHC', '1')
import mlx.core as mx

from omlx.patches.deepseek_v41 import ffn_fuse, fast_qmv
from omlx.patches.deepseek_v41.activation import quantize_swiglu_activation
from omlx.patches.deepseek_v41.loading import _load_shard
from omlx.patches.deepseek_v41.quantization import QuantizedProjection, quantize_activation

HOME = Path.home()
MODEL = HOME/'models/DeepSeek-V4.1-Flash-pipe1-mlx'
OUT = open(str(HOME/'llm/ds41/ffn/shared.jsonl'), 'a')


def emit(**r):
    line = json.dumps(r); OUT.write(line + '\n'); OUT.flush(); print(line, flush=True)


def variant_source(sg_count):
    src = ffn_fuse._SHARED_UP.replace('ROUTER\n', '')
    half = sg_count // 2
    rps = 64 // sg_count
    src = src.replace('const bool second = sg >= 8;', f'const bool second = sg >= {half};')
    src = src.replace('const int base_row = grp * 32 + (sg % 8) * 4;', f'const int base_row = grp * 32 + (sg % {half}) * {rps};')
    src = src.replace('float result[4] = {0, 0, 0, 0};\n        for (int k = 0; k < K; k += 256)',
                      f'float result[{rps}] = {{0}};\n        for (int k = 0; k < K; k += 256)')
    src = src.replace('for (int row = 0; row < 4; row++) {\n                const int out_row',
                      f'for (int row = 0; row < {rps}; row++) {{\n                const int out_row')
    src = src.replace('for (int row = 0; row < 4; row++) {\n            const float v = simd_sum(result[row]);\n            if (simd_lid == 0) out[(sg % 8) * 4 + row]',
                      f'for (int row = 0; row < {rps}; row++) {{\n            const float v = simd_sum(result[row]);\n            if (simd_lid == 0) out[(sg % {half}) * {rps} + row]')
    src = src.replace('constexpr int R = 2;', f'constexpr int R = {rps // 2};')
    src = src.replace('out[v * 32 + (sg % 8) * 4 + half_id * R + r]', f'out[v * 32 + (sg % {half}) * {rps} + half_id * R + r]')
    assert src.count(f'sg % {half}') >= 3, src
    return src


def main():
    wm = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    raw = json.loads((MODEL/'config.json').read_text())
    specs = raw['omlx_deepseek_v41']['quantized_modules']
    layers = []
    for i in (20, 21, 22):
        ns = {}
        for name in ('w1', 'w3', 'w2'):
            key = f'language_model.layers.{i}.ffn.shared_experts.{name}'
            v = _load_shard(MODEL/wm[key + '.weight'])
            ns[name] = QuantizedProjection(v[key + '.weight'], v[key + '.scales'], **specs[key])
        ns['_limit'] = 10.0
        layers.append(types.SimpleNamespace(**ns))
    mx.eval([[p.weight, p.scales] for L in layers for p in (L.w1, L.w3, L.w2)])
    kernels = {sgc: mx.fast.metal_kernel(name=f'shared_up_sg{sgc}', input_names=['x', 'sw1', 'ss1', 'sw3', 'ss3', 'limit'],
                                         output_names=['ys'], header=ffn_fuse._qmv8_header(full=True) + ffn_fuse._ROUND,
                                         source=variant_source(sgc)) for sgc in (16, 32)}

    def run_variant(sgc, L, xq):
        m, k = xq.shape
        n = L.w1.weight.shape[0]
        return kernels[sgc](inputs=[xq, L.w1.weight, L.w1.scales, L.w3.weight, L.w3.scales, ffn_fuse._limit(10.0)],
                            template=[('T', xq.dtype), ('K', k), ('N', n), ('M', m), ('RTG', 0)],
                            grid=(32 * (n // 32), sgc, 1), threadgroup=(32, sgc, 1),
                            output_shapes=[(m, n)], output_dtypes=[xq.dtype])[0]

    def reference(L, xq):
        g = L.w1.project_quantized(xq[None])[0]
        u = L.w3.project_quantized(xq[None])[0]
        return quantize_swiglu_activation(g, u, None, xq.dtype, 10.0)

    tiny = mx.array(1e-30, mx.bfloat16)
    for m in (1, 2, 3, 5):
        x = (mx.random.normal((m, 5120)) * 1.3).astype(mx.bfloat16)
        xq = quantize_activation(x)
        ref = reference(layers[0], xq)
        outs = {sgc: run_variant(sgc, layers[0], xq) for sgc in kernels}
        mx.eval(ref, outs)
        emit(test='bitwise', m=m, **{f'sg{k}': bool(mx.array_equal(ref, v).item()) for k, v in outs.items()})

        def chain(kind):
            def run():
                acc = None
                for c in range(60):
                    L = layers[c % 3]
                    xx = xq if acc is None else xq + acc
                    y = reference(L, xx) if kind == 'ref' else (xx[:, :1] if kind == 'glue' else run_variant(kind, L, xx))
                    acc = y[:, :1] * tiny
                return acc
            return run
        specs_ = ['glue', 'ref', 16, 32]
        t = {s: [] for s in specs_}
        for rep in range(12):
            for s in (specs_ if rep % 2 == 0 else specs_[::-1]):
                fn = chain(s)
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(fn())
                if rep >= 2:
                    t[s].append((time.perf_counter() - t0) * 1000)
        emit(test='chain', m=m, per_call_us={str(k): round(statistics.median(v) / 60 * 1000, 1) for k, v in t.items()})


main()
