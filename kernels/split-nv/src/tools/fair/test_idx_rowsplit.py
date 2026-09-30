"""CPU test: split_nv.idx_rowsplit == the full dense_indexer_topk call, bitwise, on both ranks.

Two TP ranks are threads sharing a fake all_gather_into_tensor (barrier + slots). For sglang's dense_indexer_topk
and for split_nv.idx_lowmem's (the production order: rowsplit(lowmem)):
  1. both ranks' gathered [rows, topk] selections equal the full call bitwise, over multi-request chunks (split point
     inside a request, empty requests, requests without positions), uneven halves (rank 0 pads), budget tilings of
     1 row .. whole chunk, and tie-heavy rows (quantized scores, whole rows of one value, -inf holes, topk larger
     than the reachable positions); the top-k is (value desc, index asc) like topk_v2_det;
  2. each rank scored and reduced exactly its own contiguous half (split at a multiple of 4), and every row's
     logits as its top-k saw them equal the full call's (int32 view);
  3. passthrough (inner called unchanged, no collective) when the chunk is not split, width <= MIN_WIDTH,
     rows < MIN_ROWS, TP != 2, or the call publishes / consumes candidate blocks;
  4. install order: idx_lowmem then idx_rowsplit wraps lowmem; idempotent; lowmem after rowsplit raises;
  5. plumbing: Front.prefill puts rank 0's decision (SPLIT_NV_IDX_ROWSPLIT and the live flag idx_rowsplit) into
     every prefill_chunk command, and Engine.execute holds it for the chunk's forward only (reset on errors too).
usage: CUDA_VISIBLE_DEVICES= python tools/fair/test_idx_rowsplit.py [path to dsv41_indexer_select.py]"""
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import types

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGS_DIR = tempfile.mkdtemp(prefix="split-nv-rowsplit-test-")
os.environ["SPLIT_NV_DIR"] = FLAGS_DIR  # perf_flags reads SPLIT_NV_DIR/box-perf-flags.json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(HERE, "..", "..", "hooks"))
import torch  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "sglang", "sglang", "srt", "layers", "attention", "dsv4",
                                                         "dsv41_indexer_select.py")
spec = importlib.util.spec_from_file_location("dsv41_indexer_select", SRC)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
from split_nv import idx_lowmem, idx_rowsplit as R, perf_flags  # noqa: E402

class RecordingPool:
    """CPU stand-in for idx_lowmem.TilePool (fixed-size tiles), per-thread like the real routing (the ranks are threads)."""

    def __init__(self):
        self.local = threading.local()
        self.cap, self.tiles, self.violations = 0, 0, []

    def active(self, tile_bytes):
        return True

    @contextlib.contextmanager
    def scope(self, device):
        self.local.inside = True
        try:
            yield
        finally:
            self.local.inside = False

    def reserve(self, device, budget_bytes):
        self.cap = max(self.cap, budget_bytes)
        self.local.reserved = True

    def note(self, logits):
        if not getattr(self.local, "reserved", False) or logits.untyped_storage().nbytes() > self.cap:
            self.violations.append("tile before reservation or above the cap")
        self.tiles += 1


POOL = RecordingPool()
INNERS = {"sglang": S.dense_indexer_topk, "lowmem": idx_lowmem.make(S), "lowmem_fixed": idx_lowmem.make(S, POOL)}
fails = []


def check(ok, what):
    if not ok:
        fails.append(what)
        print("FAIL", what)


class Group:
    def __init__(self, rank, world=2):
        self.rank_in_group, self.world_size = rank, world


def det_topk(logits, cl, k):
    """(value desc, index asc) over columns < cl, like topk_v2_det; -1 where fewer reachable than k."""
    lg = logits.clone()
    lg[torch.arange(lg.shape[1])[None, :] >= cl[:, None]] = -torch.inf
    order = torch.sort(-lg, dim=-1, stable=True).indices[:, :k]
    vals = lg.gather(1, order)
    return torch.where(vals > -torch.inf, order, torch.full_like(order, -1))


def make_case(seed, q_lens, lcs, width, ties):
    g = torch.Generator().manual_seed(seed)
    n = sum(q_lens)
    full = torch.randn(n, width, generator=g)
    if ties:
        full = (full * 2).round() / 2  # ~10 distinct values per row
        full[:: 3] = 0.5  # whole rows of one value: the top-k is decided by index alone
    full[torch.rand(n, width, generator=g) < 0.02] = -torch.inf
    comp = []
    for b, q in enumerate(q_lens):
        lc = lcs[b]
        comp += [max(1, min(lc, lc - q + 1 + i)) if lc else 0 for i in range(q)]
    compress_lens = torch.tensor(comp, dtype=torch.int32)
    ks = torch.tensor(sum(([sum(lcs[:b])] * q for b, q in enumerate(q_lens)), []), dtype=torch.int32)
    return full, compress_lens, ks


def call_kwargs(full, compress_lens, ks, q_lens, lcs, width, topk, budget_rows, log):
    """dense_indexer_topk keyword arguments; log records (scored row ranges, per-row logits seen by the top-k)."""

    def score_rows(rows):
        log["scored"].append((rows.start, rows.stop))
        return full[rows].clone()

    def topk_rows(logits, rows, out):
        for i, r in enumerate(range(rows.start, rows.stop)):
            log["seen"][r] = logits[i].clone()
        idx = det_topk(logits, compress_lens[rows].long(), topk)
        out.fill_(-1)
        sel = torch.where(idx >= 0, idx + ks[rows].long()[:, None], idx)
        out[:, :sel.shape[1]] = sel.to(out.dtype)

    return dict(score_rows=score_rows, topk_rows=topk_rows, num_tokens=sum(q_lens), width=width,
                compress_lens=compress_lens, ks=ks, q_lens_cpu=list(q_lens), lc_per_req=list(lcs), topk=topk,
                budget_bytes=budget_rows * width * 4)


def new_log():
    return {"scored": [], "seen": {}}


def run_two_ranks(inner, kwargs_for_rank):
    """Both ranks through idx_rowsplit.make(inner) in threads; returns (results, gather inputs)."""
    bar = threading.Barrier(2)
    slots, results, errors = [None, None], [None, None], []

    def gather(rank):
        def all_gather(out, inp):
            slots[rank] = inp.clone()
            bar.wait()
            out.copy_(torch.cat(slots))
            bar.wait()
        return all_gather

    def body(rank):
        try:
            fn = R.make(inner, tp=lambda: Group(rank), all_gather=gather(rank))
            results[rank] = fn(**kwargs_for_rank(rank))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            bar.abort()

    th = [threading.Thread(target=body, args=(r,)) for r in (0, 1)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errors:
        raise errors[0]
    return results, slots


# ---- 1 + 2: split == full, bitwise ---------------------------------------------------------------------------------
R.MIN_WIDTH, R.MIN_ROWS = 0, 8  # small synthetic shapes
SHAPES = (  # (q_lens, lc per request, width)
    ([37], [300], 300),  # h = 16, rank 1 21 rows: rank 0 pads
    ([64, 33], [257, 190], 260),  # split point inside request 0
    ([128], [1000], 1000),  # even halves
    ([5, 7, 0, 11], [70, 0, 33, 129], 132),  # empty request, a request without positions
    ([3, 6], [40, 20], 40),  # 9 rows: h = 4
    ([48, 1, 48], [96, 7, 700], 700),  # a one-row request right at the split
)
cases = split_cases = 0
with R.chunk(True):
    for name, inner in INNERS.items():
        for seed in range(8):
            for q_lens, lcs, width in SHAPES:
                for budget_rows in (1, 3, 16, 1000):
                    for topk in (32, 512):
                        ties = seed % 2 == 0
                        full, cl, ks = make_case(seed, q_lens, lcs, width, ties)
                        ref_log = new_log()
                        ref, pub = inner(**call_kwargs(full, cl, ks, q_lens, lcs, width, topk, budget_rows, ref_log))
                        logs = [new_log(), new_log()]
                        res, slots = run_two_ranks(inner, lambda r: call_kwargs(full, cl, ks, q_lens, lcs, width, topk,
                                                                                budget_rows, logs[r]))
                        n = sum(q_lens)
                        h = R.split_point(n)
                        tag = f"{name} seed={seed} q={q_lens} budget={budget_rows} topk={topk}"
                        cases += 1
                        split_cases += 1
                        check(h % 4 == 0 and 0 < h <= n - h, f"split point {h} of {n}")
                        for r in (0, 1):
                            sel, p = res[r]
                            check(p is None and sel.dtype == torch.int32 and torch.equal(sel, ref), f"rank {r} selections {tag}")
                            lo, hi = (0, h) if r == 0 else (h, n)
                            rows = sorted(x for a, b in logs[r]["scored"] for x in range(a, b))
                            check(rows == list(range(lo, hi)), f"rank {r} scored rows {rows[:3]}.. != [{lo}, {hi}) {tag}")
                            check(sorted(logs[r]["seen"]) == list(range(lo, hi)), f"rank {r} top-k rows {tag}")
                            check(all(torch.equal(logs[r]["seen"][x].view(torch.int32), ref_log["seen"][x].view(torch.int32))
                                      for x in range(lo, hi)), f"rank {r} logits seen by the top-k {tag}")
                        check(slots[0].shape == slots[1].shape == (n - h, topk), f"gather shapes {tag}")
check(POOL.tiles > 0 and not POOL.violations, f"fixed tiles under the row split: {POOL.tiles} tiles, {POOL.violations[:3]}")
print(f"split == full: {split_cases} cases (sglang, idx_lowmem and idx_lowmem fixed-tile inner), {len(fails)} failures")

# ---- 3: passthrough --------------------------------------------------------------------------------------------------
R.MIN_WIDTH, R.MIN_ROWS = 32768, 1024  # production defaults


def stub_inner(calls):
    def inner(**kw):
        calls.append(kw)
        return torch.full((kw["num_tokens"], kw["topk"]), 7, dtype=torch.int32), "pub"
    return inner


def no_gather(out, inp):
    raise AssertionError("collective issued")


def passthrough(on, n, width, world=2, publish=None, consume=None):
    calls = []
    fn = R.make(stub_inner(calls), tp=lambda: Group(0, world), all_gather=no_gather)
    kw = dict(score_rows=None, topk_rows=None, num_tokens=n, width=width, compress_lens=torch.zeros(n, dtype=torch.int32),
              ks=torch.zeros(n, dtype=torch.int32), q_lens_cpu=[n], lc_per_req=[width], topk=512, budget_bytes=1 << 30,
              publish_blocks=publish, consume_blocks=consume)
    with R.chunk(on):
        try:
            sel, pub = fn(**kw)
        except AssertionError:
            return False
    return len(calls) == 1 and calls[0] == kw and pub == "pub" and sel.shape == (n, 512)


pt = {"chunk not split": passthrough(False, 4096, 65536), "width == MIN_WIDTH": passthrough(True, 4096, 32768),
      "rows < MIN_ROWS": passthrough(True, 1020, 65536), "TP 1": passthrough(True, 4096, 65536, world=1),
      "publish": passthrough(True, 4096, 65536, publish=(4, 64)), "consume": passthrough(True, 4096, 65536, consume=([], 64))}
for k, v in pt.items():
    check(v, f"passthrough: {k}")
check(not passthrough(True, 4096, 32772), "a split call (width 32772, 4096 rows) issues the all-gather")
check(R._on is False, "chunk() resets the switch")
print(f"passthrough: {sum(pt.values())}/{len(pt)}; width > MIN_WIDTH and rows >= MIN_ROWS split")

# ---- 4: install order (fake sglang package around the real indexer-select module) ------------------------------------
names = ["sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.attention", "sglang.srt.layers.attention.dsv4"]
mods = {nm: types.ModuleType(nm) for nm in names}
B = types.ModuleType("sglang.srt.layers.attention.deepseek_v4_backend")
mods[B.__name__] = B
mods["sglang.srt.layers.attention.dsv4.dsv41_indexer_select"] = S
for nm, m in mods.items():
    parent, _, child = nm.rpartition(".")
    if parent in mods:
        setattr(mods[parent], child, m)
saved = {nm: sys.modules.get(nm) for nm in mods}
sys.modules.update(mods)
try:
    B.dense_indexer_topk = S.dense_indexer_topk
    idx_lowmem.install()
    R.install()
    wrapped = B.dense_indexer_topk
    check(getattr(wrapped.rowsplit_inner, "lowmem", False), "rowsplit wraps idx_lowmem")
    R.install()
    check(B.dense_indexer_topk is wrapped, "install is idempotent")
    # lowmem after rowsplit must refuse (it would drop the wrapper)
    B.dense_indexer_topk = S.dense_indexer_topk
    R.install()
    try:
        idx_lowmem.install()
        check(False, "idx_lowmem.install() after idx_rowsplit.install() raises")
    except RuntimeError:
        pass
    check(B.dense_indexer_topk.rowsplit_inner is S.dense_indexer_topk, "refused lowmem install leaves the wrapper")
finally:
    for nm, m in saved.items():
        if m is None:
            sys.modules.pop(nm, None)
        else:
            sys.modules[nm] = m
print("install order: checked")

# ---- 5: plumbing -------------------------------------------------------------------------------------------------------
from split_nv import front as F, engine as EM  # noqa: E402
from split_nv.fair import PrefillGate  # noqa: E402

F.assemble_parts = lambda *a, **k: ({}, {})


def fake_front():
    f = F.Front.__new__(F.Front)
    f.prefill_lock = threading.RLock()
    f.gate = PrefillGate(f.prefill_lock, 0, 0.5)
    f.identity, f.token_map = "t", None
    f.share_chunk, f.pf_need, f.preempt_on, f.trim_min, f.min_resume = 2048, {}, False, 0, 1024
    f.contended = lambda sid: False
    f.admit = lambda n1, sid=None: None
    f.cmds = []

    class Job:
        def wait(self):
            return {}, 0.0

    def submit_async(cmd, priority=0):
        f.cmds.append(cmd)
        return Job()

    f.submit_async = submit_async
    f.submit = lambda cmd, priority=0: None
    return f


def chunk_cmds(env, flags):
    os.environ["SPLIT_NV_IDX_ROWSPLIT"] = env
    path = os.path.join(FLAGS_DIR, "box-perf-flags.json")
    if flags is None:
        if os.path.exists(path):
            os.unlink(path)
    else:
        with open(path, "w") as fh:
            json.dump(flags, fh)
    perf_flags._state["t"] = -1e9  # re-read now
    f = fake_front()
    f.prefill(1, list(range(20001)), use_cache=False)
    return [c for c in f.cmds if c[0] == "prefill_chunk"]


for env, flags, want in (("0", None, None), ("0", {"idx_rowsplit": True}, None), ("1", None, True),
                         ("1", {"idx_rowsplit": True}, True), ("1", {"idx_rowsplit": False}, None)):
    cmds = chunk_cmds(env, flags)
    got = {c[6] if len(c) > 6 else None for c in cmds}
    check(len(cmds) == 3 and got == {want} and all(len(c) == 5 or c[5] is False for c in cmds),
          f"front commands env={env} flags={flags}: {[c[4:] for c in cmds]}")
os.environ.pop("SPLIT_NV_IDX_ROWSPLIT")

torch.cuda.synchronize = lambda *a: None  # CPU-only
E = EM.Engine.__new__(EM.Engine)
E.cap = types.SimpleNamespace(tokens=[], ntok=0, chunks=[], enabled=False)
E._cap_use = lambda sid: None
E.sessions = {5: EM.Session(5)}
seen_on = []


def extend_split(sess, chunk, split, replace_fn):
    seen_on.append(R._on)
    if len(chunk) == 3:
        raise RuntimeError("injected")


E._extend_split = extend_split
E.execute(("prefill_chunk", 5, [1, 2], None, 0, False, True))
E.execute(("prefill_chunk", 5, [1, 2], None, 0, False, False))
E.execute(("prefill_chunk", 5, [1, 2], None, 0))
E.execute(("prefill_chunk", 5, [1, 2], None, 0, False))
try:
    E.execute(("prefill_chunk", 5, [1, 2, 3], None, 0, False, True))
except RuntimeError:
    pass
check(seen_on == [True, False, False, False, True] and R._on is False, f"engine switch per chunk: {seen_on}, after {R._on}")
print("plumbing: checked")

print(f"{cases} split cases + passthrough/install/plumbing checks; {len(fails)} failures")
assert not fails, fails
print("PASS")
