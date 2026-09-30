"""DSpark-on-box wire protocol v1 (STEPD / STPD / RING, OPEN/ACK capability): codecs and the pure decision rules.

The normative text is /home/ian/mac/pi-ds41-runs/v2/STEPD-SPEC.md (STATUS: FROZEN v1). Everything here is plain
Python (struct/json/hashlib), no torch, so the Mac side can port it verbatim and CPU tests can drive it.
All integers little-endian; BF16 = raw IEEE bfloat16 bits (u16 LE); F32 = IEEE binary32 LE.
"""
import hashlib
import json
import math
import struct

VER = 1
TAP_DIM = 15360  # 3 target layers x 5120 BF16: the Mac's DSpark main_proj input per committed position
KEY_DIM = 512  # DSpark head_dim (K = V, MQA)
STAGES = 3
RING = 128  # committed positions per stage ring (slot = position % RING)
WIDTH = 4  # drafted tokens per box draft (block = anchor + WIDTH)
BLOCK = 5
MIN_CTX = 1024  # the box drafts (mode 0) only at keep >= MIN_CTX
MAX_ROWS = 8  # a step's L (STEP_API: graph widths up to 8)
MAX_EXPLICIT = MAX_ROWS - 1
ROW_BYTES = 4 * 5120 * 2 + 4 * 4 + 288 + 68  # one STPR/STPD payload row (h19 | pre | ckv20 | idxk20)
TAP_ROW_BYTES = TAP_DIM * 2
KEY_ROW_BYTES = KEY_DIM * 2

# envelope tags ('<4sIQ'): the STEPD request travels as b"DSTP" (SPEC errata E1: "STEPD" has 5 bytes)
TAG_STEPD, TAG_STPD, TAG_RING = b"DSTP", b"STPD", b"RING"
STEPD_HDR = struct.Struct("<IIIBBBBBB")  # session, keep, anchor, nver, a, mode, dmax, flags, n_explicit (18 B)
STPR_HDR = struct.Struct("<IIHIf")  # session, length_after, L, payload_bytes, box_seconds (18 B)
STPD_EXT = struct.Struct("<BBBBBBHf4I4f")  # a_box, depth, mode_used, flags, reason, ndraft, reserved, drafter_ms, drafts, maxprob
STPD_HDR_SIZE = STPR_HDR.size + STPD_EXT.size  # 62 B

# STEPD.mode / STPD.mode_used
MODE_BOX, MODE_EXPLICIT, MODE_BONUS = 0, 1, 2
# STEPD.flags
F_FUSED = 1 << 0  # fused regime (og_fused live): the box uses the "fused" cost table
F_NO_TAPS = 1 << 1  # nver > 0 but no taps attached (the box ring gets a gap: ring_ok -> 0)
# STPD.flags
R_DRAFTED = 1 << 0  # this step's drafts were drafted on the box
R_PF_ACTIVE = 1 << 1  # another session's prefill is running or queued (fairness hint, SPEC s7)
R_RING_OK = 1 << 2  # the box ring is current through keep (after this job)
R_DISABLED = 1 << 3  # box drafting is off (runtime kill switch)
# STPD.reason (why mode_used differs from the request, or why the ring is not ok)
REASONS = {0: "none", 1: "disabled", 2: "ring_not_ok", 3: "short_context", 4: "dmax_zero", 5: "ring_rejected"}
REASON = {v: k for k, v in REASONS.items()}
A_BOX_NONE = 255  # STPD.a_box for a kickoff (nver == 0)


class WireError(ValueError):
    """A hard protocol error: ERR {error, retry: false, code} and the session is closed (as for STEP)."""

    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code


# ---------------------------------------------------------------------------------------------- OPEN / ACK
def parse_costs(obj):
    """OPEN.dspark.costs -> {"pipe": ((floor, (c2, c3, c4, c5)), ...), "fused": ...}, tiers sorted by floor
    descending (the Mac's _pipeline_costs order). Raises ValueError on anything malformed."""
    if not isinstance(obj, dict) or set(obj) != {"pipe", "fused"}:
        raise ValueError("costs needs exactly the keys pipe and fused")
    out = {}
    for name in ("pipe", "fused"):
        tiers = obj[name]
        if not isinstance(tiers, list) or not 1 <= len(tiers) <= 8:
            raise ValueError(f"costs.{name}: 1..8 tiers")
        norm = []
        for t in tiers:
            if not (isinstance(t, list) and len(t) == 2 and isinstance(t[0], int) and not isinstance(t[0], bool)
                    and t[0] >= 0 and isinstance(t[1], list) and len(t[1]) == 4):
                raise ValueError(f"costs.{name}: tier must be [floor >= 0, [c2, c3, c4, c5]]")
            cs = tuple(float(c) for c in t[1])
            if any(not math.isfinite(c) or c <= 0 for c in cs):
                raise ValueError(f"costs.{name}: costs must be finite and > 0")
            norm.append((int(t[0]), cs))
        floors = [f for f, _ in norm]
        if 0 not in floors or len(set(floors)) != len(floors):
            raise ValueError(f"costs.{name}: needs a floor-0 tier and distinct floors")
        out[name] = tuple(sorted(norm, reverse=True))
    return out


def parse_open(h):
    """OPEN header's dspark request -> (costs, None) or (None, reason string) or (None, None) if not requested."""
    req = h.get("dspark")
    if req is None:
        return None, None
    if not isinstance(req, dict) or req.get("ver") != VER:
        return None, "version"
    try:
        return parse_costs(req.get("costs")), None
    except ValueError:
        return None, "bad_request"


def capability():
    """ACK.dspark when granted. step_taps / filler: SPEC v2 additions V2.2 / V2.3 (announced, wire ver stays 1)."""
    return {"ver": VER, "width": WIDTH, "block": BLOCK, "ring": RING, "min_ctx": MIN_CTX, "tap_dim": TAP_DIM,
            "key_dim": KEY_DIM, "stages": STAGES, "max_explicit": MAX_EXPLICIT, "step_taps": 1, "filler": 1}


# ---------------------------------------------------------------------------------------------- cost policy
def tier(table, context):
    """The costs (c2..c5) of the first tier with context >= floor (tiers sorted by floor descending)."""
    return next(c for floor, c in table if context >= floor)


def choose_cost_depth(probs, costs, max_depth=WIDTH):
    """Mac PipelineDepthController.choose_cost_depth (context >= 1024), verbatim arithmetic (Python floats)."""
    cumulative, expected, best, best_utility = 1.0, 1.0, 1, -1.0
    for index, p in enumerate(list(probs)[:max_depth]):
        cumulative *= max(0.0, min(1.0, float(p)))
        expected += cumulative
        utility = expected / costs[index]
        if utility > best_utility:
            best, best_utility = index + 1, utility
    return best


def box_depth(probs, costs_by_regime, keep, fused, dmax):
    """Width of a box-drafted step: min(choose_cost_depth(maxprob[0:4], tier(table, keep)), dmax); the caller
    drafts only with dmax >= 1 (and caps dmax to the context: keep + 1 + dmax <= context length)."""
    table = costs_by_regime["fused" if fused else "pipe"]
    return max(1, min(choose_cost_depth(probs, tier(table, keep)), dmax))


# ---------------------------------------------------------------------------------------------- accept
def accept_count(ids_prev, argmax):
    """a_box: the first i in [0, nver-1) with argmax[i] != ids_prev[i+1], else nver-1."""
    n = len(ids_prev)
    for i in range(n - 1):
        if argmax[i] != ids_prev[i + 1]:
            return i
    return n - 1


def check_accept(pending, keep, anchor, nver, a, argmax):
    """STEPD with nver > 0 against the box's pending step (base, ids_prev). Returns a_box or raises WireError."""
    if pending is None:
        raise WireError("no_pending", f"nver={nver} but the session has no pending step (kickoff after OPEN/reopen: nver=0)")
    base, ids_prev = pending
    if len(ids_prev) != nver:
        raise WireError("accept_mismatch", f"nver={nver} but the pending step has {len(ids_prev)} rows")
    if keep != base + a + 1:
        raise WireError("accept_mismatch", f"keep={keep} != base {base} + a {a} + 1")
    a_box = accept_count(ids_prev, argmax)
    if a_box != a or argmax[a] != anchor:
        raise WireError("accept_mismatch", f"a={a} anchor={anchor}: box a_box={a_box}, argmax[a]={argmax[a]}")
    return a_box


# ---------------------------------------------------------------------------------------------- STEPD
def encode_stepd(session, keep, anchor, nver, a, mode, dmax, flags=0, argmax=(), explicit=(), taps=b""):
    """-> (header 18 B, payload). taps: raw BF16 bytes [a+1, TAP_DIM] (b"" with nver == 0 or F_NO_TAPS)."""
    hdr = STEPD_HDR.pack(session, keep, anchor, nver, a, mode, dmax, flags, len(explicit))
    payload = struct.pack(f"<{nver}I", *argmax) + struct.pack(f"<{len(explicit)}I", *explicit) + bytes(taps)
    return hdr, payload


def decode_stepd(header, payload, max_pos=1 << 32):
    """Frame-level validation (no session state). Returns a dict; taps is a memoryview of the payload."""
    if len(header) != STEPD_HDR.size:
        raise WireError("bad_stepd", f"header {len(header)} bytes (v1: {STEPD_HDR.size})")
    session, keep, anchor, nver, a, mode, dmax, flags, n_explicit = STEPD_HDR.unpack(header)
    if mode not in (MODE_BOX, MODE_EXPLICIT, MODE_BONUS):
        raise WireError("bad_stepd", f"mode {mode}")
    if dmax > WIDTH:
        raise WireError("bad_stepd", f"dmax {dmax} > {WIDTH}")
    if n_explicit and mode != MODE_EXPLICIT:
        raise WireError("bad_stepd", f"n_explicit {n_explicit} with mode {mode}")
    if n_explicit > MAX_EXPLICIT:
        raise WireError("bad_stepd", f"n_explicit {n_explicit} > {MAX_EXPLICIT}")
    if nver > MAX_ROWS:
        raise WireError("bad_stepd", f"nver {nver} > {MAX_ROWS}")
    if nver == 0 and a != 0:
        raise WireError("bad_stepd", "nver=0 needs a=0")
    if nver and a >= nver:
        raise WireError("bad_stepd", f"a {a} >= nver {nver}")
    ntaps = (a + 1) if nver and not flags & F_NO_TAPS else 0
    want = 4 * nver + 4 * n_explicit + ntaps * TAP_ROW_BYTES
    if len(payload) != want:
        raise WireError("bad_stepd", f"payload {len(payload)} bytes, expected {want}")
    L = 1 + n_explicit  # the rows this frame requires (box drafts are capped to the context instead)
    if keep + L > max_pos:
        raise WireError("context_exceeded", f"context length {max_pos} exceeded at {keep + L}")
    mv = memoryview(payload)
    argmax = list(struct.unpack_from(f"<{nver}I", mv, 0))
    explicit = list(struct.unpack_from(f"<{n_explicit}I", mv, 4 * nver))
    return {"session": session, "keep": keep, "anchor": anchor, "nver": nver, "a": a, "mode": mode, "dmax": dmax,
            "flags": flags, "argmax": argmax, "explicit": explicit, "ntaps": ntaps,
            "taps": mv[4 * (nver + n_explicit):]}


# ---------------------------------------------------------------------------------------------- STPD
def encode_stpd_header(session, keep, L, box_s, a_box, mode_used, flags, reason, drafts=(), maxprob=(), drafter_ms=0.0):
    nd = len(drafts)
    d = list(drafts) + [0] * (WIDTH - nd)
    p = [float(x) for x in maxprob] + [0.0] * (WIDTH - len(maxprob))
    return (STPR_HDR.pack(session, keep + L, L, L * ROW_BYTES, box_s)
            + STPD_EXT.pack(a_box, L - 1, mode_used, flags, reason, nd, 0, drafter_ms, *d, *p))


def decode_stpd_header(h):
    if len(h) != STPD_HDR_SIZE:
        raise WireError("bad_stpd", f"STPD header {len(h)} bytes (v1: {STPD_HDR_SIZE})")
    session, after, L, nbytes, box_s = STPR_HDR.unpack_from(h, 0)
    v = STPD_EXT.unpack_from(h, STPR_HDR.size)
    a_box, depth, mode_used, flags, reason, nd, _, drafter_ms = v[:8]
    return {"session": session, "length_after": after, "L": L, "payload_bytes": nbytes, "box_s": box_s,
            "a_box": a_box, "depth": depth, "mode_used": mode_used, "flags": flags, "reason": reason, "ndraft": nd,
            "drafter_ms": drafter_ms, "drafts": list(v[8:8 + WIDTH]), "maxprob": list(v[8 + WIDTH:8 + 2 * WIDTH])}


# ---------------------------------------------------------------------------------------------- RING
def ring_bytes(keys_rows, taps_rows):
    return STAGES * keys_rows * KEY_ROW_BYTES + taps_rows * TAP_ROW_BYTES


def encode_ring(session, offset, keys, taps, keys_rows, taps_rows, digest=True):
    """keys: BF16 bytes [STAGES, keys_rows, KEY_DIM] (per stage oldest first); taps: BF16 bytes [taps_rows, TAP_DIM].
    -> (JSON header bytes, payload)."""
    payload = bytes(keys) + bytes(taps)
    h = {"session": session, "offset": offset, "keys_rows": keys_rows, "taps_rows": taps_rows}
    if digest:
        h["sha256"] = hashlib.sha256(payload).hexdigest()
    return json.dumps(h, separators=(",", ":")).encode(), payload


def decode_ring(h, payload, max_pos=1 << 32):
    """Frame-level validation of a RING (JSON header dict). Returns a dict with memoryviews keys / taps and
    digest_ok (None when no sha256 was sent). Geometry errors are hard; a digest mismatch is soft (ring_rejected)."""
    try:
        session, offset = int(h["session"]), int(h["offset"])
        kr, tr = int(h.get("keys_rows", 0)), int(h.get("taps_rows", 0))
    except (KeyError, TypeError, ValueError):
        raise WireError("bad_ring", f"RING header {str(h)[:200]}")
    n = kr + tr
    if kr < 0 or tr < 0 or not 1 <= n <= RING or offset < n or offset > max_pos:
        raise WireError("bad_ring", f"RING rows keys {kr} + taps {tr} (1..{RING}) at offset {offset}")
    if len(payload) != ring_bytes(kr, tr):
        raise WireError("bad_ring", f"RING payload {len(payload)} bytes, expected {ring_bytes(kr, tr)}")
    digest_ok = None
    if h.get("sha256") is not None:
        digest_ok = hashlib.sha256(payload).hexdigest() == str(h["sha256"]).lower()
    mv = memoryview(payload)
    kb = STAGES * kr * KEY_ROW_BYTES
    return {"session": session, "offset": offset, "keys_rows": kr, "taps_rows": tr, "keys": mv[:kb], "taps": mv[kb:],
            "digest_ok": digest_ok}
