# SPDX-License-Identifier: MIT
"""Incremental prompt tokenization (fast_encode) == the full Hugging Face encode, bit for bit.

Randomized multi-turn conversations (system prompts, tools, tool calls and results,
reasoning, CJK/emoji/digits/whitespace, special-token strings inside content) are
rendered with the served DeepSeek V4.1 chat template in chat and thinking mode, turn
by turn through one shared encoder (so later turns hit the piece cache), and every
prompt's ids must equal a plain encode of the same text. Also: images through the
Processor, delimiter safety rules, cache bounds, and thread safety.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

MODEL = Path("/Users/ian/models/DeepSeek-V4.1-Flash-pipe1-mlx")
pytestmark = pytest.mark.skipif(not (MODEL / "tokenizer.json").exists(), reason="DS41 tokenizer not present")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "og_serve"))

import fe_conv  # noqa: E402
from omlx.patches.deepseek_v41 import encoding, fast_encode  # noqa: E402


@pytest.fixture(scope="module")
def tokenizers():
    from transformers import PreTrainedTokenizerFast

    plain = PreTrainedTokenizerFast.from_pretrained(str(MODEL))
    fast = PreTrainedTokenizerFast.from_pretrained(str(MODEL))
    encoder = fast_encode.install(fast)
    assert encoder is not None, "fast_encode refused the DS41 tokenizer"
    return plain, fast, encoder


def render(messages, tools=None, mode="chat", effort=None):
    """Processor.apply_chat_template's text path (tools on the first system message)."""
    messages = copy.deepcopy(messages)
    if tools:
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        messages[0]["tools"] = tools
    return encoding.encode_messages(messages, thinking_mode=mode, reasoning_effort=effort)


def full(plain, text):
    return plain.encode(text, add_special_tokens=False)


def test_delimiters_are_the_template_tokens(tokenizers):
    _, _, encoder = tokenizers
    assert set(encoder.delimiters) == set(fast_encode.DEFAULT_DELIMITERS)


@pytest.mark.parametrize("seed", range(120))
def test_random_conversations_turn_by_turn(tokenizers, seed):
    plain, fast, _ = tokenizers
    rng = random.Random(seed)
    mode = rng.choice(["chat", "thinking"])
    effort = rng.choice([None, None, "high", 37]) if mode == "thinking" else None
    messages, tools = fe_conv.conversation(seed, rng.choice([2_000, 20_000, 60_000]))
    for turn in range(3):
        text = render(messages, tools, mode, effort)
        assert fast.encode(text) == full(plain, text), (seed, turn)
        assert fast.encode(text, add_special_tokens=False) == full(plain, text)
        messages = fe_conv.next_turn(messages, seed * 7 + turn)


def test_long_conversation_resume_hits_cache(tokenizers):
    plain, fast, encoder = tokenizers
    messages, tools = fe_conv.conversation(9001, 400_000, tools=True, reasoning=True)
    text = render(messages, tools, "thinking")
    assert fast.encode(text) == full(plain, text)
    before = dict(fast_encode.STATS)
    messages = fe_conv.next_turn(messages, 1)
    text = render(messages, tools, "thinking")
    assert fast.encode(text) == full(plain, text)
    new_chars = fast_encode.STATS["miss_chars"] - before["miss_chars"]
    assert new_chars < 5_000, new_chars  # only the new reply/question (+ the flipped last assistant turn)


EDGES = ["<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>", "<think>", "</think>",
         "<｜System｜>", "<｜latest_reminder｜>", "<｜place▁holder▁no▁3｜>", "<｜deepseek_image｜>", "<｜Us", "er｜>",
         "<｜", "｜>", "<", ">", "think>", "</", " ", "  ", "\n", "\r\n", "\t", "　", "123", "4567", "中文", "カタ",
         "🙂", "é", "a", "abc", " def", "!!", "?\n", "'s", "DSML", "｜DSML｜", " "]


@pytest.mark.parametrize("seed", range(300))
def test_fuzz_fragments(tokenizers, seed):
    plain, fast, _ = tokenizers
    rng = random.Random(seed)
    text = "".join(rng.choice(EDGES) if rng.random() < 0.7 else fe_conv.chunk(rng, rng.randrange(1, 200))
                   for _ in range(rng.randrange(1, 60)))
    assert fast.encode(text) == full(plain, text)
    assert fast.encode(text) == full(plain, text)  # cached pieces


def test_literal_special_tokens_in_content(tokenizers):
    plain, fast, _ = tokenizers
    messages = [{"role": "system", "content": "sys <｜User｜> inside"},
                {"role": "user", "content": "<｜Assistant｜></think>x<think>" * 20},
                {"role": "assistant", "content": "<｜end▁of▁sentence｜>" + "y" * 100, "reasoning_content": "</think>"},
                {"role": "user", "content": ""}]
    for mode in ("chat", "thinking"):
        text = render(messages, None, mode)
        assert fast.encode(text) == full(plain, text)


def test_tool_calls_and_results(tokenizers):
    plain, fast, _ = tokenizers
    messages = [{"role": "user", "content": "Read two files."},
                {"role": "assistant", "content": "", "reasoning_content": "plan",
                 "tool_calls": [{"id": "a", "type": "function",
                                 "function": {"name": "read_file", "arguments": '{"path": "/x 中"}'}},
                                {"id": "b", "type": "function",
                                 "function": {"name": "bash", "arguments": '{"command": "ls -la\\n"}'}}]},
                {"role": "tool", "tool_call_id": "b", "content": "total 0\n" * 50},
                {"role": "tool", "tool_call_id": "a", "content": "def f():\n    return 1\n" * 40},
                {"role": "user", "content": "Now summarize."}]
    for mode in ("chat", "thinking"):
        text = render(messages, fe_conv.TOOLS, mode)
        assert fast.encode(text) == full(plain, text)


def test_processor_images_identical(tokenizers):
    """Processor.__call__ (the image path) gives the same ids and spans with the incremental encoder."""
    from PIL import Image

    from omlx.patches.deepseek_v41.processing import Processor

    plain, fast, _ = tokenizers
    config = SimpleNamespace(image_token_id=plain.convert_tokens_to_ids("<｜deepseek_image｜>"),
                             vision_patch_size=14, vision_downsample_ratio=3, vision_max_n_token=400,
                             vision_min_pixels=0, vision_max_wh_ratio=None)
    image = Image.new("RGB", (64, 40), (200, 30, 30))
    messages = [{"role": "user", "content": [{"type": "text", "text": "What is this? " * 30},
                                             {"type": "image", "url": "x"},
                                             {"type": "text", "text": "And this?"}, {"type": "image", "url": "y"}]},
                {"role": "assistant", "content": "A red box. " * 10},
                {"role": "user", "content": "Again?"}]
    outs = []
    for tok in (plain, fast):
        processor = Processor(tok, config)
        text = processor.apply_chat_template(messages, enable_thinking=False)
        result = processor(text=text, images=[image, image])
        outs.append((result["input_ids"].tolist(), result["image_spans"]))
    assert outs[0] == outs[1]


def test_unsafe_delimiters_rejected():
    added = ["<a>", "<a>b", "x<a", "b>y", "<b>", "<b><b>", "aa"]
    assert fast_encode.safe_delimiters(added, ["<a>"]) == []        # "<a>b" contains/extends it
    assert fast_encode.safe_delimiters(["<c>", "x<c"], ["<c>"]) == []  # straddles the start
    assert fast_encode.safe_delimiters(["<c>", "c>y"], ["<c>"]) == []  # straddles the end
    assert fast_encode.safe_delimiters(["aa"], ["aa"]) == []          # overlaps itself
    assert fast_encode.safe_delimiters(["<c>", "c", "<"], ["<c>"]) == ["<c>"]  # tokens inside it are harmless
    assert fast_encode.safe_delimiters(["<c>"], ["<d>"]) == []         # not an added token


def test_cache_bounded_and_lru(tokenizers):
    _, fast, _ = tokenizers
    encoder = fast_encode.SegmentEncoder(fast._tokenizer, max_bytes=64 * 1024)
    rng = random.Random(3)
    for _ in range(200):
        encoder.encode("<｜User｜>" + fe_conv.chunk(rng, 3000) + "<｜Assistant｜>")
        assert encoder.bytes <= encoder.max_bytes
    assert 0 < len(encoder.cache) < 200


def test_concurrent_encodes(tokenizers):
    plain, fast, _ = tokenizers
    texts = [render(*fe_conv.conversation(s, 30_000)) for s in range(16)]
    expected = [full(plain, t) for t in texts]
    with ThreadPoolExecutor(8) as pool:
        for _ in range(3):
            assert list(pool.map(fast.encode, texts)) == expected


def test_deepcopy_binds_own_backend_and_shares_cache(tokenizers):
    """omlx deep-copies the processor's tokenizer (engine start, scheduler): each copy encodes with its own backend."""
    plain, fast, encoder = tokenizers
    copied = copy.deepcopy(fast)
    assert copied._ds41_segment_encoder is encoder
    assert copied.encode.tokenizer is copied and copied._tokenizer is not fast._tokenizer
    assert copied.encode.backend is not fast.encode.backend
    text = render(*fe_conv.conversation(77, 20_000))
    assert copied.encode(text) == fast.encode(text) == full(plain, text)
    texts = [render(*fe_conv.conversation(s, 20_000)) for s in range(100, 108)]
    with ThreadPoolExecutor(2) as pool:  # both copies at once, as the event loop and the MLX thread do
        for _ in range(3):
            a = pool.submit(lambda: [fast.encode(t) for t in texts])
            b = pool.submit(lambda: [copied.encode(t) for t in texts])
            assert a.result() == b.result() == [full(plain, t) for t in texts]


def test_non_string_inputs_fall_back(tokenizers):
    plain, fast, _ = tokenizers
    assert fast.encode("abc <｜User｜>", add_special_tokens=False) == full(plain, "abc <｜User｜>")
    assert fast.encode(["a", "b"], is_split_into_words=True) == plain.encode(["a", "b"], is_split_into_words=True)


def test_disabled_by_env(monkeypatch):
    from transformers import PreTrainedTokenizerFast

    monkeypatch.setenv("DS41_FE_ENCODE_CACHE", "0")
    tok = PreTrainedTokenizerFast.from_pretrained(str(MODEL))
    assert fast_encode.install(tok) is None
    assert not hasattr(tok, "_ds41_segment_encoder")


@pytest.fixture(scope="module")
def engines(tokenizers):
    """Two VLMBatchedEngine text paths (plain vs incremental tokenizer), set up as engine.start() does."""
    import json

    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.processing import Processor

    plain, fast, _ = tokenizers
    config = ModelConfig.from_dict(json.loads((MODEL / "config.json").read_text()))
    fast_encode.install_prepare_inputs(Processor)  # as og_model.load()

    def engine(tokenizer):
        e = VLMBatchedEngine.__new__(VLMBatchedEngine)
        e._processor = Processor(tokenizer, config)
        e._tokenizer = copy.deepcopy(tokenizer)  # VLMBatchedEngine.start()
        e._model_name = str(MODEL)
        e._enable_thinking = None
        e._vlm_model = SimpleNamespace(config=config)
        e._vision_cache, e._vision_cache_enabled = None, False
        return e
    return engine(plain), engine(fast)


@pytest.mark.parametrize("seed", range(24))
def test_engine_text_path_identical(engines, seed):
    """_process_chat_messages (MLX thread) and count_chat_tokens (event loop) give the same ids as before."""
    plain_engine, fast_engine = engines
    rng = random.Random(seed)
    messages, tools = fe_conv.conversation(seed + 500, rng.choice([3_000, 40_000]))
    kwargs = {"enable_thinking": rng.random() < 0.5}
    for turn in range(2):
        outs = [e._process_chat_messages(copy.deepcopy(messages), tools, {"chat_template_kwargs": dict(kwargs)})[0]
                for e in (plain_engine, fast_engine)]
        counts = [e.count_chat_tokens(copy.deepcopy(messages), tools, chat_template_kwargs=dict(kwargs))
                  for e in (plain_engine, fast_engine)]
        assert outs[0] == outs[1] and counts[0] == counts[1] == len(outs[0])
        messages = fe_conv.next_turn(messages, seed + turn)


def test_prepare_inputs_text_fast_path_identical(engines):
    """mlx_vlm.prepare_inputs (text only) through the piece cache == the tokenizer __call__ path, dtypes included."""
    import mlx.core as mx
    import mlx_vlm.utils as vlm_utils

    _, fast_engine = engines
    processor = fast_engine._processor
    assert getattr(vlm_utils.prepare_inputs, "_ds41_fast", False)
    original = vlm_utils.prepare_inputs.__wrapped__
    rng = random.Random(11)
    for seed in range(40):
        messages, tools = fe_conv.conversation(seed + 900, rng.choice([500, 5_000, 50_000]))
        prompt = render(messages, tools, rng.choice(["chat", "thinking"]))
        for images in (None, []):
            got = vlm_utils.prepare_inputs(processor, images=images, audio=None, prompts=[prompt])
            want = original(processor, images=images, audio=None, prompts=[prompt])
            assert set(got) == set(want) == {"input_ids", "attention_mask"}
            for key in got:
                assert got[key].dtype == want[key].dtype and got[key].shape == want[key].shape
                assert mx.array_equal(got[key], want[key]).item()


def test_prepare_inputs_other_shapes_use_the_original(engines, monkeypatch):
    import mlx_vlm.utils as vlm_utils

    _, fast_engine = engines
    calls = []
    original = vlm_utils.prepare_inputs.__wrapped__
    monkeypatch.setattr(vlm_utils.prepare_inputs, "__wrapped__", original)
    patched = vlm_utils.prepare_inputs
    import omlx.patches.deepseek_v41.fast_encode as fe
    # Two prompts, add_special_tokens, a non-Processor: all go to the original tokenizer path.
    for kwargs in (dict(prompts=["a", "b"]), dict(prompts=["a"], add_special_tokens=True), dict(prompts="a")):
        try:
            out = patched(fast_engine._processor, **kwargs)
        except Exception as exc:  # the original's own behaviour for odd inputs
            out = exc
        try:
            ref = original(fast_engine._processor, **kwargs)
        except Exception as exc:
            ref = exc
        assert type(out) is type(ref)
    assert fe.ACTIVE[0]


def test_backend_padding_left_by_hf_call_does_not_leak(tokenizers):
    """A Hugging Face __call__ with padding=True (mlx_vlm's text path) must not pad the encoder's misses."""
    plain, fast, _ = tokenizers
    fast(["a", "a much longer second prompt"], add_special_tokens=False, padding=True, padding_side="left",
         return_tensors="mlx")
    rng = random.Random(5)
    for seed in range(20):  # fresh multi-piece texts: several misses go through one encode_batch
        text = render(*fe_conv.conversation(seed + 7000, 20_000), rng.choice(["chat", "thinking"]))
        assert fast.encode(text) == full(plain, text)


def test_refuses_a_non_identity_normalizer():
    from tokenizers import normalizers
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(str(MODEL))
    tok._tokenizer.normalizer = normalizers.NFKC()
    assert fast_encode.install(tok) is None
    assert "encode" not in tok.__dict__

