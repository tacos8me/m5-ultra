# Copyright © 2026 Apple Inc.

import copy
import os
import unittest

# Exact fp32 matmuls, so that verify and decode steps pick the same tokens.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx.utils import tree_map

from mlx_lm.generate import generate_step, speculative_generate_step
from mlx_lm.models import mimo_v2
from mlx_lm.models.cache import make_prompt_cache


def make_mimo(num_mtp=3, window=4, vocab_size=8):
    args = mimo_v2.ModelArgs.from_dict(
        {
            "model_type": "mimo_v2",
            "vocab_size": vocab_size,
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "v_head_dim": 16,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 32,
            "swa_v_head_dim": 16,
            "swa_rope_theta": 1000.0,
            "sliding_window_size": window,
            "add_full_attention_sink_bias": False,
            "add_swa_attention_sink_bias": True,
            "hybrid_layer_pattern": [0, 1, 1],
            "moe_layer_freq": [0, 1, 1],
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "topk_method": "noaux_tc",
            "partial_rotary_factor": 0.5,
            "attention_bias": False,
            "layernorm_epsilon": 1e-6,
            "max_position_embeddings": 1000,
            "rope_theta": 10000.0,
            "num_nextn_predict_layers": num_mtp,
        }
    )
    model = mimo_v2.Model(args)
    model.update(
        tree_map(lambda p: mx.random.normal(p.shape) * 0.5, model.parameters())
    )
    mx.eval(model.parameters())
    return model


class TestMTP(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)
        self.model = make_mimo()

    def greedy(self, prompt, max_tokens):
        return [t for t, _ in generate_step(prompt, self.model, max_tokens=max_tokens)]

    def test_sanitize_drops_missing_mtp(self):
        model = make_mimo()
        weights = {"lm_head.weight": mx.zeros((8, 64))}
        model.sanitize(weights)
        self.assertIsNone(model.model.mtp)
        with self.assertRaises(ValueError):
            model.make_draft_model()

    def test_greedy_matches_target(self):
        draft = self.model.make_draft_model()
        prompt = mx.random.randint(0, 8, (19,))
        expected = self.greedy(prompt, 48)
        for num_draft in [1, 2, 3, 4]:
            out = list(
                speculative_generate_step(
                    prompt,
                    self.model,
                    draft,
                    num_draft_tokens=num_draft,
                    max_tokens=48,
                )
            )
            self.assertEqual([t for t, _, _ in out], expected)
            self.assertTrue(any(d for _, _, d in out))

    def test_reuse_prompt_cache(self):
        draft = self.model.make_draft_model()
        n = len(self.model.layers)
        cache = make_prompt_cache(self.model) + make_prompt_cache(draft)
        for size in [11, 5, 1]:
            prompt = mx.random.randint(0, 8, (size,))
            ref_cache = copy.deepcopy(cache[:n])
            expected = [
                t
                for t, _ in generate_step(
                    prompt, self.model, prompt_cache=ref_cache, max_tokens=20
                )
            ]
            out = speculative_generate_step(
                prompt, self.model, draft, prompt_cache=cache, max_tokens=20
            )
            self.assertEqual([t for t, _, _ in out], expected)

    def test_new_draft_cache_after_prompt_cache(self):
        # The server can pair a reused model cache with a new draft cache.
        draft = self.model.make_draft_model()
        cache = make_prompt_cache(self.model)
        first = mx.random.randint(0, 8, (30,))
        self.model(first[None], cache=cache)
        prompt = mx.random.randint(0, 8, (6,))
        ref_cache = copy.deepcopy(cache)
        expected = [
            t
            for t, _ in generate_step(
                prompt, self.model, prompt_cache=ref_cache, max_tokens=20
            )
        ]
        cache += make_prompt_cache(draft)
        out = speculative_generate_step(
            prompt, self.model, draft, prompt_cache=cache, max_tokens=20
        )
        self.assertEqual([t for t, _, _ in out], expected)
        self.assertGreater(cache[-1].layers[0].offset, first.size)


if __name__ == "__main__":
    unittest.main()
