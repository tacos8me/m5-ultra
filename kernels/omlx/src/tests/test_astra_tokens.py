"""Exact byte comparison retains the digest and invalid-token gates."""
import hashlib,struct
from types import SimpleNamespace
import pytest
import mlx.core as mx
mx.set_default_device(mx.cpu)
from omlx.patches.deepseek_v41 import pipe_decoder
# Fail before tensors are imported; this does not allocate GPU arrays or a model.
def reach_tokens(tokens, raw_tokens, digest_tokens=None, dtype='U32'):
 n=len(tokens);data=struct.pack('<%dI'%len(raw_tokens),*raw_tokens)
 c=SimpleNamespace(n_layers=40)
 man=dict(format=pipe_decoder.FORMAT,identity='same',prompt_tokens=n,
          token_sha256=hashlib.sha256(struct.pack('<%dI'%n,*(tokens if digest_tokens is None else digest_tokens))).hexdigest(),layers=[{}]*21)
 tensors={'tokens':(dtype,[len(raw_tokens)],memoryview(data),len(data))}
 return pipe_decoder.DecoderHalf.import_state_steps(SimpleNamespace(_config=c),tensors,man,tokens,identity='same')
@pytest.mark.parametrize('bad',[[1,3],[1],[1,2,3],[2,1]])
def test_wrong_wire_tokens(bad):
 with pytest.raises(ValueError,match='tokens mismatch'):next(reach_tokens([1,2],bad))
def test_wrong_digest():
 with pytest.raises(ValueError,match='prompt mismatch'):next(reach_tokens([1,2],[1,2],[1,3]))
def test_wrong_kind():
 with pytest.raises(ValueError,match='Unexpected encoder tensor'):next(reach_tokens([1,2],[1,2],dtype='I32'))
@pytest.mark.parametrize('ids', [[0,2**32-1],[1,128799],[True,2]])
def test_matching_bytes_reach_next_layout_check(ids):
 with pytest.raises(AttributeError,match='kv_source_layers'):next(reach_tokens(ids,ids))
