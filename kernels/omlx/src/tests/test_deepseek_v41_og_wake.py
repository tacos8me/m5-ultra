# SPDX-License-Identifier: MIT
"""og_model: a box OPEN that finishes mid-step is admitted by the next step, not after the idle wait."""


def test_open_finishing_during_a_step_is_admitted_without_the_idle_wait():
    """og_model.opened_step/opened_wake: an OPEN done mid-step makes the step report work."""
    from omlx.patches.deepseek_v41 import og_model

    class Output:
        has_work = False

    class Sched:
        woken = 0

        def _ds41_wake(self):
            self.woken += 1

    sched = Sched()
    wake = og_model.opened_wake(sched)

    def step_with_open(self):
        wake()  # the Job thread finishes its OPEN while the step runs
        return Output()

    def idle_step(self):
        return Output()

    assert og_model.opened_step(step_with_open)(sched).has_work is True and sched.woken == 1
    assert og_model.opened_step(idle_step)(sched).has_work is False
