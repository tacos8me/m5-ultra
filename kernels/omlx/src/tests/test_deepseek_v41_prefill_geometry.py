from types import SimpleNamespace
from omlx.scheduler import Scheduler
from omlx.patches.deepseek_v41.language import LanguageModel
from test_deepseek_v41 import tiny, tiny_ced


def test_model_prefill_geometry_is_specific_to_ced():
    assert LanguageModel(tiny_ced(ced_prefill=True))._omlx_preserve_prefill_chunks
    assert not LanguageModel(tiny(ced_prefill=False))._omlx_preserve_prefill_chunks


def test_contention_keeps_declared_prefill_geometry():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._decode_fairness = True
    scheduler._decode_contention = lambda: True
    scheduler._prefill_tps_best = 2000.0
    scheduler.config = SimpleNamespace(prefill_step_size=8192)
    for name in ("_language_model", "language_model"):
        scheduler.model = SimpleNamespace(
            **{name: SimpleNamespace(_omlx_preserve_prefill_chunks=True)}
        )
        assert scheduler._contended_prefill_cap() == 0
    scheduler.model = SimpleNamespace(_omlx_preserve_prefill_chunks=True)
    assert scheduler._contended_prefill_cap() == 0
    scheduler.model = SimpleNamespace()
    assert 0 < scheduler._contended_prefill_cap() < 8192
