"""CPU-only tests: pipe_wire session rebuild (tier A) against the fake box.

Run: python og_serve/test_wire_failover.py   (no MLX, no GPU, no real box)
"""
import importlib.util
import os
from pathlib import Path
import struct
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('DS41_OG_RESUME_WAIT_S', '6')
os.environ.setdefault('DS41_OG_RESTART_WAIT_S', '6')  # the fake box's down() refuses the port
os.environ.setdefault('DS41_OG_STEP_TIMEOUT_S', '5')
os.environ.setdefault('DS41_OG_SPIN_MS', '0')
spec = importlib.util.spec_from_file_location(
    'pipe_wire', os.environ.get('PIPE_WIRE', str(HERE.parent/'omlx/patches/deepseek_v41/pipe_wire.py')))
wire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wire)
from fakebox import FakeBox, digest  # noqa: E402

PORT = int(os.environ.get('FAKEBOX_PORT', '12169'))


def rows_ok(raw, expected_prefixes):
    for i, tokens in enumerate(expected_prefixes):
        if raw[i*wire.ROW_BYTES:i*wire.ROW_BYTES+32] != digest(tokens):
            return False
    return True


class Client:
    """Mirror of og_model's use: ensure_step_safe then recv_step_safe, tracking the true sequence."""

    def __init__(self, prompt):
        self.enc = wire.EncoderSession('127.0.0.1', PORT)
        self.enc.open(prompt)
        self.seq = list(prompt[:-1])

    def step(self, ids, keep):
        self.enc.ensure_step_safe(ids, keep)
        raw, _ = self.enc.recv_step_safe(ids, keep)
        expected = [self.seq[:keep] + ids[:i+1] for i in range(len(ids))]
        assert rows_ok(raw, expected), 'box rows do not match the committed sequence'
        self.seq = self.seq[:keep] + ids
        return raw


def case(name, fn):
    t0 = time.time()
    try:
        fn()
        print(f'PASS {name} ({time.time()-t0:.1f}s)', flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f'FAIL {name}: {exc!r}', flush=True)
        return False


def main():
    box = FakeBox(PORT)
    prompt = list(range(100, 140))

    def normal():
        c = Client(prompt)
        c.step([prompt[-1], 7, 8, 9], 39)       # kickoff + 3 drafts, all accepted
        c.step([10, 11], 41)                    # rollback: keep 2 of the 4 rows
        c.step([12], 43)
        assert not c.enc.recoveries

    def killed_mid_step():
        c = Client(prompt)
        c.step([prompt[-1], 5, 6], 39)
        c.enc.send_step([6, 20, 21], 41)        # pre-sent step whose reply is lost
        box.kill()
        c.step([6, 20, 21], 41)                 # finds the pending step dead, rebuilds at 41, resends
        assert len(c.enc.recoveries) == 1 and c.enc.recoveries[0]['keep'] == 41
        opens = [e for e in box.log if e[0] == 'open' and e[3] == 'none']
        assert opens and opens[-1][2] == 42, opens[-1:]
        c.step([21, 30], 42)

    def stale_presend_on_dead_link():
        c = Client(prompt)
        c.step([prompt[-1], 5], 39)
        c.enc.send_step([5, 9, 9], 40)          # pre-sent, then the loop issues different rows
        box.kill()
        c.step([5, 8], 40)
        c.step([8, 1], 41)

    def restart_within_wait():
        c = Client(prompt)
        c.step([prompt[-1], 5], 39)
        box.down()
        threading.Timer(2.5, box.up).start()
        t0 = time.monotonic()
        c.step([5, 6, 7], 40)
        assert 2.0 < time.monotonic() - t0 < 6, time.monotonic() - t0
        c.step([7], 42)

    def down_past_wait():
        c = Client(prompt)
        c.step([prompt[-1], 5], 39)
        box.down()
        try:
            c.step([5, 6], 40)
        except wire.BoxLost as exc:
            assert 'box unavailable' in str(exc)
        else:
            raise AssertionError('expected BoxLost')
        finally:
            box.up()

    def refusing_box():
        c = Client(prompt)
        c.step([prompt[-1], 5], 39)
        box.err_opens = True
        box.kill()
        t0 = time.monotonic()
        try:
            c.step([5, 6], 40)
        except wire.BoxLost:
            assert time.monotonic() - t0 < 5, 'a refusing box should fail after RESUME_TRIES, not the wait'
        else:
            raise AssertionError('expected BoxLost')
        finally:
            box.err_opens = False

    def busy_box_waits():
        c = Client(prompt)
        c.step([prompt[-1], 5], 39)
        box.err_opens = 'busy'
        box.kill()
        threading.Timer(2.5, lambda: setattr(box, 'err_opens', False)).start()
        t0 = time.monotonic()
        c.step([5, 6], 40)
        assert time.monotonic() - t0 > 2, 'a busy box must be waited for, not counted as a failed try'

    def two_sessions_one_lost():
        a, b = Client(prompt), Client(prompt[:20] + [1, 2, 3])
        a.step([prompt[-1]], 39)
        b.step([3], 22)
        a.enc.send_step([50, 51], 40)
        b.enc.send_step([60], 23)
        # kill only a's connection (the box drops one session)
        a.enc.sock.shutdown(2)
        a.step([50, 51], 40)
        b.step([60], 23)
        assert a.enc.recoveries and not b.enc.recoveries

    ok = all([case('normal steps and rollback', normal),
              case('connection killed with a step in flight', killed_mid_step),
              case('stale pre-sent step on a dead link', stale_presend_on_dead_link),
              case('box restarts within the wait', restart_within_wait),
              case('box down past the wait -> BoxLost', down_past_wait),
              case('box refuses reopen -> BoxLost after tries', refusing_box),
              case('busy box (retryable ERR) is waited for', busy_box_waits),
              case('c2: one session lost, the other untouched', two_sessions_one_lost)])
    box.down()
    print('ALL PASS' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
