"""Prefill q without the extra copy (SPLIT_NV_Q_NOCOPY=1, default off).

MQALayer.forward (dsv41) hands _compute_q_b a persistent q buffer (q_padded, [rows, heads, 512]) that the attention
kernel reads; _compute_q_b runs wq_b into a fresh tensor, applies RoPE in place and then copies the whole thing into
that buffer: at 8192 rows 268 MB per layer, ~6 ms per 8K chunk. When the buffer has exactly q's shape (prefill on
SM120: no head padding), wq_b's MXFP8 GEMM can write into it directly (flashinfer mm_mxfp8 takes `out`) and the copy
goes away. Same GEMM kernel and tactic, same RoPE kernel, same bytes in the buffer the attention reads.
"""
import os

_pending = [None]  # output buffer for the next MXFP8 GEMM call (set only around wq_b)
ACTIVE = True  # tests switch it at runtime; the engine leaves it on once installed


def enabled():
    return os.environ.get("SPLIT_NV_Q_NOCOPY", "0") == "1"


def install():
    from sglang.srt.layers.quantization import fp8_utils as F
    from sglang.srt.models import deepseek_v4 as M

    if getattr(M.MQALayer, "_q_nocopy", False):
        return
    mm = F.flashinfer_mm_mxfp8
    raw = F._raw_flashinfer_mm_mxfp8

    def mm_out(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype, use_8x4_sf_layout=False, backend="auto"):
        out = _pending[0]
        _pending[0] = None
        if (out is not None and backend == "cutlass" and out.dtype == out_dtype
                and out.shape == (q_input.shape[0], weight_t.shape[1])):
            # the b12x kernel the hooks route this shape to, writing into the caller's buffer
            try:
                return raw(q_input, weight_t, x_scale_u8, weight_scale_t, out=out, out_dtype=out_dtype,
                           use_8x4_sf_layout=use_8x4_sf_layout, backend="b12x")
            except Exception:  # noqa: BLE001  (shape b12x rejects: the normal path, and the caller copies)
                pass
        return mm(q_input, weight_t, x_scale_u8, weight_scale_t, out_dtype=out_dtype,
                  use_8x4_sf_layout=use_8x4_sf_layout, backend=backend)

    F.flashinfer_mm_mxfp8 = mm_out
    orig = M.MQALayer._compute_q_b

    def _compute_q_b(self, q, positions, q_out=None):
        rows = q.shape[0]
        if (not ACTIVE or q_out is None or self.q_head_norm or not q_out.is_contiguous()
                or tuple(q_out.shape) != (rows, self.n_local_heads, self.head_dim)):
            return orig(self, q, positions, q_out)
        _pending[0] = q_out.view(rows, -1)
        try:
            qq, _ = self.wq_b(q)
        finally:
            _pending[0] = None
        qq = qq.view(-1, self.n_local_heads, self.head_dim)
        M.fused_rope_inplace(qq[..., -self.qk_rope_head_dim:], None, self.freqs_cis, positions=positions)
        if qq.data_ptr() != q_out.data_ptr():  # the GEMM did not take the buffer: copy as before
            q_out.copy_(qq)
        return q_out

    M.MQALayer._compute_q_b = _compute_q_b
    M.MQALayer._q_nocopy = True
