"""Repetitive prompts for the topk-det window gates (their indexer threshold bins overflow; TOPK-DET.md section 2).

Writes plain id lists [BOS, <User>, ...] like ref/ids-8192.json to /mnt/nvme-1/split-nv-ops/topk/ids/:
  ids-line-<n>.json   one nginx access-log line repeated verbatim
  ids-code-<n>.json   one ~420-token Python function repeated verbatim
usage: python3 make_rep_ids.py [n ...]   (default 262144)
"""
import json
import os
import sys

from transformers import AutoTokenizer

CK = "/home/ian/models/DeepSeek-V4.1-Flash-original"
OUT = "/mnt/nvme-1/split-nv-ops/topk/ids"
HEAD = [0, 128803]  # BOS, <｜User｜> (as in ref/ids-8192.json)

LINE = ('10.0.1.17 - - [29/Sep/2026:00:12:07 +0000] "GET /api/v1/items/48213 HTTP/1.1" 200 5123 "-" '
        '"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36" '
        'rt=0.042 uct="0.000" uht="0.041"\n')

CODE = '''def merge_intervals(intervals: list[tuple[int, int]], *, min_gap: int = 0) -> list[tuple[int, int]]:
    """Merge overlapping or nearly-adjacent half-open intervals.

    Intervals closer than ``min_gap`` are merged as well. The input may be unsorted and may contain
    empty intervals, which are dropped. Returns a new sorted list; the input is not modified.
    """
    if min_gap < 0:
        raise ValueError(f"min_gap must be non-negative, got {min_gap}")
    cleaned = sorted((start, end) for start, end in intervals if end > start)
    if not cleaned:
        return []
    merged: list[tuple[int, int]] = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= min_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    total = sum(end - start for start, end in merged)
    logger.debug("merged %d intervals into %d covering %d units", len(intervals), len(merged), total)
    return merged


'''


def main():
    tok = AutoTokenizer.from_pretrained(CK)
    os.makedirs(OUT, exist_ok=True)
    for n in [int(x) for x in sys.argv[1:]] or [262144]:
        for name, text in (("line", LINE), ("code", CODE)):
            ids = tok.encode(text, add_special_tokens=False)
            body = (ids * (n // len(ids) + 1))[: n - len(HEAD)]
            path = f"{OUT}/ids-{name}-{n}.json"
            json.dump(HEAD + body, open(path, "w"))
            print(path, len(HEAD) + len(body), "period", len(ids))


if __name__ == "__main__":
    main()
