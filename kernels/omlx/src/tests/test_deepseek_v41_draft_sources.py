"""CPU-only contracts for optional proposal sources."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

path = Path(__file__).resolve().parents[1] / 'omlx/patches/deepseek_v41/draft_sources.py'
spec = importlib.util.spec_from_file_location('draft_sources_tested', path)
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)

class Tokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))
    def decode(self, ids):
        return ''.join(map(chr, ids))

SCHEMAS = [{'name': 'read_file', 'parameters': {'properties': {'path': {'type': 'string'}, 'line': {'type': 'integer'}}}}]

def test_bank_only_has_verified_complete_suffixes():
    bank = ds.Bank()
    feature = np.array([1., 0.], np.float32)
    bank.add(b'a', [10, 11, 12, 13], [(0, feature)])
    assert bank.propose(b'a', 10, feature) is None
    bank.add(b'a', [10, 11, 12, 13, 14, 15], [(0, feature)])
    assert bank.propose(b'a', 10, feature) == [11, 12, 13, 14]
    assert bank.propose(b'b', 10, feature) is None
    assert bank.propose(b'a', 10, np.array([0., 1.])) is None
    feature[:] = 0
    assert bank.propose(b'a', 10, np.array([1., 0.])) == [11, 12, 13, 14]

def test_bank_storage_is_bounded():
    bank = ds.Bank(); feature = np.array([1., 0.], np.float32)
    for anchor in range(140):
        for _ in range(66):
            bank.add(b'a', [anchor, 1, 2, 3, 4], [(0, feature)])
    assert len(bank.rows) == 128
    assert sum(map(len, bank.rows.values())) == 8192

def test_schema_proposal_and_budget():
    tok = Tokenizer(); tracker = ds.Tracker(tok, SCHEMAS, ds.Bank(), 1, 'schema')
    text = '<｜DSML｜ invoke name="read_file">\n<｜DSML｜ parameter name="path" string="true">'
    prefix = '<｜DSML｜ invoke name="read_file">\n'
    draft, source = tracker.propose(tok.encode(prefix), 4)
    assert source == 'schema'
    assert draft == tok.encode(text[len(prefix):len(prefix)+4])
    assert tracker.propose(tok.encode(prefix), 0) == (None, None)
    assert tracker.propose(tok.encode('unrelated text'), 4) == (None, None)

def test_finish_clamps_to_emitted_tokens_and_is_once_only():
    bank = ds.Bank(); tracker = ds.Tracker(Tokenizer(), SCHEMAS, bank, 1, 'both')
    tracker.tokens = list(range(10))
    tracker.features = [(i, np.array([1., 0.])) for i in range(10)]
    tracker.finish(5)
    assert sum(map(len, bank.rows.values())) == 1
    tracker.finish(10)
    assert sum(map(len, bank.rows.values())) == 1

def test_create_requires_enabled_mode_and_real_tool_schemas(monkeypatch):
    tok = Tokenizer(); host = SimpleNamespace(_ds41_draft_tokenizer=tok)
    prompt = '### Available Tool Schemas\n\n' + json.dumps(SCHEMAS[0]) + '\n\nUse the tools.\nX'
    ids = tok.encode(prompt); copy = SimpleNamespace(_buf=np.array(ids), n=len(ids))
    monkeypatch.setattr(ds, 'MODE', 'off'); assert ds.create(host, copy) is None
    monkeypatch.setattr(ds, 'MODE', 'both'); assert ds.create(host, copy) is not None
    assert ds.create(host, SimpleNamespace(_buf=np.array(tok.encode('helloX')), n=6)) is None


def test_filter_only_publishes_complete_tool_calls():
    tok = Tokenizer(); bank = ds.Bank()
    tracker = ds.Tracker(tok, SCHEMAS, bank, 1, 'both')
    tracker.tokens = tok.encode('partial call')
    tracker.features = [(0, np.array([1., 0.]))]
    tracker.finish(len(tracker.tokens), tool_filter=True)
    assert not tracker.done and not bank.rows
    tracker.tokens += tok.encode('</｜DSML｜ calls>')
    tracker.finish(len(tracker.tokens), tool_filter=True)
    assert tracker.done and sum(map(len, bank.rows.values())) == 1


def test_scheduler_filter_finishes_removed_uid_before_reindexing():
    # Run the actual patched-filter body with a CPU-only generation-batch double.
    import ast
    source = path.parents[1] / 'mlx_lm_mtp/batch_generator.py'
    node = next(n for n in ast.walk(ast.parse(source.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == 'patched_filter')
    priming = SimpleNamespace(release_uids=lambda *a: None)
    def original_filter(batch, keep, *args, **kwargs):
        batch.uids = [batch.uids[i] for i in keep]
        batch._num_tokens = [batch._num_tokens[i] for i in keep]
        return 'filtered'
    namespace = dict(original_filter=original_filter, _prompt_priming=priming,
                     _drop_invalid_mtp_state=lambda *a, **kw: None,
                     _drop_invalid_mtp_batch_state=lambda *a, **kw: None,
                     _mtp_park_state_for_batch=lambda *a: None)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    bank = ds.Bank(); tok = Tokenizer()
    complete = ds.Tracker(tok, SCHEMAS, bank, 1, 'both')
    pending = ds.Tracker(tok, SCHEMAS, bank, 1, 'both')
    complete.tokens = tok.encode('prefix</｜DSML｜ calls>')
    pending.tokens = tok.encode('partial')
    complete.features = pending.features = [(0, np.array([1., 0.]))]
    states = {'a': SimpleNamespace(extra_source_tracker=complete),
              'b': SimpleNamespace(extra_source_tracker=pending)}
    batch = SimpleNamespace(uids=['a', 'b'], model=None,
                            _num_tokens=[len(complete.tokens), len(pending.tokens)],
                            _omlx_mtp_batch_state=SimpleNamespace(states=states))
    assert namespace['patched_filter'](batch, [1]) == 'filtered'
    assert batch.uids == ['b'] and complete.done and not pending.done
    assert sum(map(len, bank.rows.values())) == 1
    namespace['patched_filter'](batch, [])
    assert not pending.done and sum(map(len, bank.rows.values())) == 1
