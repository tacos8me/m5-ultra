# Copyright © 2026 Apple Inc.

import importlib
import math
import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import speculative_generate_step
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_sampler, speculative_sample


class BigramLM(nn.Module):
    """A model whose next-token logits depend only on the last token."""

    def __init__(self, table):
        super().__init__()
        self.table = table

    def __call__(self, inputs, cache=None):
        if cache is not None:
            x = mx.zeros((1, 1, inputs.shape[1], 1))
            cache[0].update_and_fetch(x, x)
        return self.table[inputs]

    def make_cache(self):
        return [KVCache()]


def chi2_pvalue(stat, df):
    # Wilson-Hilferty approximation of the chi-square upper tail.
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def transition_test(tokens, probs):
    """Chi-square test of the transition counts of ``tokens`` against ``probs``."""
    V = probs.shape[0]
    counts = mx.zeros((V, V), dtype=mx.int32)
    counts = counts.at[mx.array(tokens[:-1]), mx.array(tokens[1:])].add(1)
    counts, probs = counts.tolist(), probs.tolist()
    stat, df, impossible = 0.0, 0, 0
    for row_counts, row_probs in zip(counts, probs):
        n = sum(row_counts)
        support = [p > 0 for p in row_probs]
        if n == 0:
            continue
        df += sum(support) - 1
        for c, p, s in zip(row_counts, row_probs, support):
            if s:
                stat += (c - n * p) ** 2 / (n * p)
            else:
                impossible += c
    return chi2_pvalue(stat, df), impossible


def resample_from_target(draft_tokens, draft_logits, target_logits):
    # Broken on purpose: after a rejection it samples p instead of max(0, p - q).
    num_accepted, _ = speculative_sample(draft_tokens, draft_logits, target_logits)
    return num_accepted, mx.random.categorical(target_logits)[num_accepted]


class TestSpeculativeSampling(unittest.TestCase):
    V = 6

    def setUp(self):
        mx.random.seed(0)
        target = mx.random.normal((self.V, self.V)) * 1.5
        # A draft that is close to the target but not equal.
        draft = target + mx.random.normal((self.V, self.V))
        self.target = BigramLM(target)
        self.draft = BigramLM(draft)

    def target_probs(self, sampler):
        logprobs = nn.log_softmax(self.target.table, axis=-1)
        return mx.softmax(sampler.logits(logprobs), axis=-1)

    def generate(self, sampler, num_draft, num_tokens=6000):
        mx.random.seed(1)
        out = speculative_generate_step(
            mx.array([0]),
            self.target,
            self.draft,
            sampler=sampler,
            num_draft_tokens=num_draft,
            max_tokens=num_tokens,
        )
        tokens, drafted = [0], 0
        for t, _, from_draft in out:
            tokens.append(t)
            drafted += from_draft
        return tokens, drafted / num_tokens

    def test_speculative_sample_first_token(self):
        # The first output token of one verify step has the target distribution.
        mx.random.seed(2)
        k, n = 3, 3000
        p_logits = mx.random.normal((k + 1, self.V)) * 2
        q_logits = p_logits[:k] + mx.random.normal((k, self.V)) * 2
        counts = [0] * self.V
        accepted = 0
        for _ in range(n):
            draft = mx.random.categorical(q_logits)
            num_accepted, token = speculative_sample(draft, q_logits, p_logits)
            first = draft[0] if num_accepted.item() > 0 else token
            counts[first.item()] += 1
            accepted += num_accepted.item() > 0
        p = mx.softmax(p_logits[0]).tolist()
        stat = sum((c - n * pi) ** 2 / (n * pi) for c, pi in zip(counts, p))
        self.assertGreater(chi2_pvalue(stat, self.V - 1), 1e-4)

        # The first draft is accepted with probability sum(min(p, q)).
        q = mx.softmax(q_logits[0]).tolist()
        expected = sum(min(a, b) for a, b in zip(p, q))
        sigma = math.sqrt(expected * (1 - expected) / n)
        self.assertLess(abs(accepted / n - expected), 4 * sigma)

    def test_output_matches_target(self):
        cases = [
            (make_sampler(temp=1.0), 1),
            (make_sampler(temp=0.6, top_p=0.9), 3),
            (make_sampler(temp=1.0, top_k=3), 2),
            (make_sampler(temp=0.8, min_p=0.1), 3),
        ]
        for sampler, num_draft in cases:
            probs = self.target_probs(sampler)
            tokens, drafted = self.generate(sampler, num_draft)
            pvalue, impossible = transition_test(tokens, probs)
            self.assertEqual(impossible, 0)
            self.assertGreater(pvalue, 1e-4)
            self.assertGreater(drafted, 0.1)

    def test_broken_variant_fails(self):
        sampler = make_sampler(temp=1.0)
        probs = self.target_probs(sampler)
        module = importlib.import_module("mlx_lm.generate")
        with mock.patch.object(module, "speculative_sample", resample_from_target):
            tokens, _ = self.generate(sampler, 2)
        pvalue, _ = transition_test(tokens, probs)
        self.assertLess(pvalue, 1e-6)

    def test_sampler_logits(self):
        sampler = make_sampler(temp=0.5, top_p=0.9)
        logprobs = nn.log_softmax(mx.random.normal((1, 32)), axis=-1)
        mx.random.seed(3)
        expected = sampler(logprobs)
        mx.random.seed(3)
        self.assertEqual(
            mx.random.categorical(sampler.logits(logprobs)).item(), expected.item()
        )
        self.assertFalse(hasattr(make_sampler(temp=0.0), "logits"))


if __name__ == "__main__":
    unittest.main()
