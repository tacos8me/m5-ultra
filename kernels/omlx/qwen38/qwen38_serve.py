"""`omlx serve` with Qwen4 QSA KV caches sized once to the reserved prompt length plus an output margin.

Stock growth doubles capacity (…, 256K, 512K, 1M); each doubling keeps all 12 layers' old buffers in the MLX cache
while the new ones are filled, a ~+24 GiB spike at the 512K -> 1M step. Allocating the reservation up front avoids it.
MTP prompt priming evaluates the head cache's K/V per chunk, but never its indexer state or pooled block bank (the
head's priming output is discarded). Each chunk's pending bank update pins that chunk's index-key buffer, so every later
write copies the whole buffer; the chain is only evaluated at the first draft (~+20 GiB at 1M). The unreserved (MTP
head) cache now evaluates its indexer keys and bank after each multi-token update; the main caches are left alone.
"""
import os, sys

MARGIN = int(os.environ.get('QWEN_KV_MARGIN', '32768'))


def _install():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
    apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache

    orig = QSAKVCache.update_and_fetch

    def update_and_fetch(self, keys, values):
        reserve = getattr(self, '_index_reserved_tokens', 0)
        prev, cur = self.offset, 0 if self.keys is None else self.keys.shape[2]
        needed = prev + keys.shape[2]
        if reserve and needed > cur and reserve + MARGIN > cur:
            cap = -(-max(needed, reserve + MARGIN) // self.step) * self.step
            B, H, _, dk = keys.shape
            new_k = mx.zeros((B, H, cap, dk), keys.dtype)
            new_v = mx.zeros((B, H, cap, values.shape[3]), values.dtype)
            if prev:
                new_k[..., :prev, :] = self.keys[..., :prev, :]
                new_v[..., :prev, :] = self.values[..., :prev, :]
            self.keys, self.values = new_k, new_v
        return orig(self, keys, values)

    QSAKVCache.update_and_fetch = update_and_fetch

    stock_update_indexer = QSAKVCache.update_indexer

    def update_indexer(self, keys, position_ids):
        out = stock_update_indexer(self, keys, position_ids)
        if keys.shape[1] > 1 and not getattr(self, '_index_reserved_tokens', 0):
            mx.eval([a for a in (self.keys, self.values, self._index_keys, self._index_position_ids) if a is not None])
        return out

    QSAKVCache.update_indexer = update_indexer

    stock_pooled = QSAKVCache.pooled_indexer_keys

    def pooled_indexer_keys(self, *a, **k):
        before = self._pooled_index_offset
        out = stock_pooled(self, *a, **k)
        if self._pooled_index_offset - before > 1 and not getattr(self, '_index_reserved_tokens', 0):
            mx.eval(self._pooled_index_keys)
        return out

    QSAKVCache.pooled_indexer_keys = pooled_indexer_keys


if __name__ == '__main__':
    _install()
    from omlx.cli import main
    sys.argv[0] = 'omlx'
    sys.exit(main())
