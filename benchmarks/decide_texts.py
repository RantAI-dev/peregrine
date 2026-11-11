#!/usr/bin/env python3
"""Build the token-calibrated states for the decision benchmark.

Runs from the benchmark venv (needs `transformers`), writes JSON that the
stdlib-only runner (`decide.py`) consumes.  A state is one text whose token
count is within 2% of the requested length with that model's own tokenizer.

  python decide_texts.py --tokenizer-dir DIR --lengths 64,300,1000 \
      --out states.json
"""

import argparse
import json
from pathlib import Path
import re
import sys

from transformers import AutoTokenizer

CORPUS = Path(__file__).resolve().parent / "prompts" / "corpus.txt"


def load_words():
    text = CORPUS.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    text = " ".join(lines)
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-]*|[,.;:!?]", text)


def count(tok, words, start, k):
    return len(tok(" ".join(words[start:start + k]), add_special_tokens=True)["input_ids"])


def build(tok, words, start, target):
    """The word window whose token count is closest to `target`."""
    lo, hi = 1, min(len(words) - start, target * 3 + 8)
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if count(tok, words, start, mid) <= target:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    cand = []
    for k in (best - 2, best - 1, best, best + 1, best + 2):
        if 1 <= k <= len(words) - start:
            n = count(tok, words, start, k)
            cand.append((abs(n - target), k, n))
    _, k, n = min(cand)
    return " ".join(words[start:start + k]), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--lengths", default="64,300,1000")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x]
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    words = load_words()
    # A long stream of the same public corpus, so a 1 000-token state has room.
    reps = max(1, 6000 // len(words) + 1)
    words = (words * reps)[:6000]
    out = {"tokenizer": str(Path(args.tokenizer_dir).resolve()), "states": {}}
    for i, L in enumerate(lengths):
        text, n = build(tok, words, i * 97 + L * 3, L)
        out["states"][str(L)] = {"text": text, "tokens": n}
        print(f"state {L}: {n} tokens")
        if not (0.98 * L <= n <= 1.02 * L):
            print(f"[warn] state {L} came out at {n} tokens", file=sys.stderr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
