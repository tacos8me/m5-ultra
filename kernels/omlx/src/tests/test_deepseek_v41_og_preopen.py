# SPDX-License-Identifier: MIT
"""ds41-c4: burst admission starts the box OPEN of queued requests early (OgPrefill.preopen). CPU, no network:
Job is replaced by a recorder, the scheduler by its admission-relevant surface (waiting deque + slot counters)."""

from collections import deque
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from omlx.patches.deepseek_v41 import og_model  # noqa: E402

STARTED = []


class FakeJob:
    def __init__(self, host, port, request_id, tokens, images=None, wake=None):
        self.request_id, self.tokens, self.images = request_id, list(tokens), images
        self.done, self.error, self.cancelled, self.encoder = threading.Event(), None, False, None
        self.cancel_event = threading.Event()  # og_model.Job gained it with the cancellation fixes (ds41-astra/robust)
        self.replay = self.imported = self.replay_error = None

    def start(self):
        STARTED.append(self.request_id)


class FakeScheduler:
    def __init__(self, requests, running=0, max_num_seqs=4):
        self.waiting = deque(requests)
        self.running = [object()] * running
        self.prefilling = []
        self.max_num_seqs = max_num_seqs

    def _effective_max_num_seqs(self):
        return self.max_num_seqs

    def _num_admitted_requests(self):
        return len(self.running) + len(self.prefilling)


def req(rid, n=64, images=None):
    return SimpleNamespace(request_id=rid, prompt_token_ids=list(range(n)), vlm_inputs_embeds=images,
                           arrival_time=0.0)


@pytest.fixture(autouse=True)
def fake_jobs(monkeypatch):
    STARTED.clear()
    monkeypatch.setattr(og_model, "Job", FakeJob)
    monkeypatch.setattr(og_model, "PREOPEN", 3)
    monkeypatch.setattr(og_model, "PREOPEN_CHAIN", False)


def defer(manager, sched):
    """og_model.install's _defer order: the head's own OPEN first, then the queue behind it."""
    try:
        return manager.should_defer(sched, sched.waiting[0])
    finally:
        manager.preopen(sched)


def test_burst_opens_in_arrival_order():
    manager = og_model.OgPrefill("127.0.0.1", 1)
    sched = FakeScheduler([req("a"), req("b"), req("c"), req("d")])
    assert defer(manager, sched) is True
    assert STARTED == ["a", "b", "c", "d"]
    # the next steps find the jobs already running: no second OPEN for anyone
    assert defer(manager, sched) is True
    sched.waiting.popleft()
    defer(manager, sched)
    assert STARTED == ["a", "b", "c", "d"]


def test_free_slots_bound_the_open_sessions():
    manager = og_model.OgPrefill("127.0.0.1", 1)
    sched = FakeScheduler([req("a"), req("b"), req("c")], running=2)  # 4 slots: 2 running + head + 1
    defer(manager, sched)
    assert STARTED == ["a", "b"]
    sched.running.pop()
    defer(manager, sched)
    assert STARTED == ["a", "b", "c"]


def test_images_invalid_and_off_keep_the_head_path(monkeypatch):
    manager = og_model.OgPrefill("127.0.0.1", 1)
    sched = FakeScheduler([req("a"), req("img", images=object()), req("tiny", n=1), req("d")])
    defer(manager, sched)
    assert STARTED == ["a", "d"]  # the image request and the invalid one open (or fail) at the head
    STARTED.clear()
    monkeypatch.setattr(og_model, "PREOPEN", 0)
    manager2 = og_model.OgPrefill("127.0.0.1", 1)
    defer(manager2, FakeScheduler([req("x"), req("y")]))
    assert STARTED == ["x"]


def test_chain_opens_one_box_prefill_at_a_time(monkeypatch):
    monkeypatch.setattr(og_model, "PREOPEN_CHAIN", True)
    manager = og_model.OgPrefill("127.0.0.1", 1)
    sched = FakeScheduler([req("a"), req("b"), req("c"), req("d")])
    defer(manager, sched)
    assert STARTED == ["a"]  # the head's OPEN is still running on the box
    manager.jobs["a"].done.set()
    defer(manager, sched)
    assert STARTED == ["a", "b"]
    defer(manager, sched)
    assert STARTED == ["a", "b"]
    manager.jobs["b"].done.set()
    sched.waiting.popleft()  # a admitted meanwhile
    defer(manager, sched)
    assert STARTED == ["a", "b", "c"]
    manager.jobs["c"].done.set()
    defer(manager, sched)
    assert STARTED == ["a", "b", "c", "d"]
    # an image request queued in between keeps the chain closed until it opens at the head
    STARTED.clear()
    manager2 = og_model.OgPrefill("127.0.0.1", 1)
    sched2 = FakeScheduler([req("x"), req("img", images=object()), req("y")])
    defer(manager2, sched2)
    manager2.jobs["x"].done.set()
    defer(manager2, sched2)
    assert STARTED == ["x"]


def test_abort_of_a_preopened_request_cancels_its_open(monkeypatch):
    monkeypatch.setattr(og_model.og_failover, "clear_open", lambda rid: None)
    manager = og_model.OgPrefill("127.0.0.1", 1)
    sched = FakeScheduler([req("a"), req("b")])
    defer(manager, sched)
    job = manager.jobs["b"]
    job.done.set()
    manager.abort("b")
    assert job.cancelled and "b" not in manager.jobs
