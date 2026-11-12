#!/usr/bin/env python3
"""Build the deterministic, token-calibrated text sets for the embedding benchmark.

Runs from the benchmark venv (needs `transformers`), writes JSON that the
stdlib-only runner consumes, so every engine sees byte-identical texts for a
model.  A text's token count is calibrated with that model's own tokenizer to
within 5% of the requested length.

  python embed_texts.py --tokenizer-dir DIR --lengths 16,128,512 \
      --per-length 128 --max-tokens 510 --out texts.json
"""

import argparse
import json
from pathlib import Path
import random
import re
import sys

from transformers import AutoTokenizer

CORPUS = Path(__file__).resolve().parent / "prompts" / "corpus.txt"


def load_words(corpus_path: Path):
    text = corpus_path.read_text(encoding="utf-8")
    # Drop the leading comment lines that describe the corpus.
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    text = " ".join(lines)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’\-]*|[,.;:!?]", text)
    return words


def count(tok, words, start: int, k: int) -> int:
    return len(tok(" ".join(words[start:start + k]), add_special_tokens=True)["input_ids"])


def build_text(tok, words, start: int, target: int, max_tokens: int):
    """Grow a word window from `start` whose token count is closest to target."""
    cap = max_tokens if max_tokens else 100000
    hi_limit = min(cap, int(target * 1.05))
    lo, hi = 1, min(len(words) - start, target * 3 + 8)
    # Largest prefix within the 5% upper bound, then the closest of its
    # neighbours (token counts are monotone in word count for these tokenizers).
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if count(tok, words, start, mid) <= hi_limit:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    # An exact hit matters most for the short targets, where one token is 6%:
    # prefer it over the nearest miss, then fall back to the closest window.
    window = range(max(1, best - 8), min(len(words) - start, best + 8) + 1)
    exact = [(k, count(tok, words, start, k)) for k in window]
    for k, n in exact:
        if n == target:
            return " ".join(words[start:start + k]), n
    cand = []
    for k in (best - 2, best - 1, best, best + 1, best + 2):
        if 1 <= k <= len(words) - start:
            n = count(tok, words, start, k)
            if n <= cap:
                cand.append((abs(n - target), k, n))
    if not cand:
        k = 1
        return " ".join(words[start:start + k]), count(tok, words, start, k)
    _, k, n = min(cand)
    # If the closest window is short of the target, start from the next larger
    # one so the character trim below can cut *down* to the exact count.
    if n < target and k + 1 <= len(words) - start and count(tok, words, start, k + 1) <= cap:
        k, n = k + 1, count(tok, words, start, k + 1)
    text = " ".join(words[start:start + k])
    # Word granularity alone can miss a short target (a 15-word prefix is 17
    # tokens, 16 is unreachable); trim the tail by characters to hit it exactly.
    text, n = trim_chars(tok, text, target, cap)
    return text, n


def trim_chars(tok, text: str, target: int, cap: int):
    lo, hi = 1, len(text)
    full_n = len(tok(text, add_special_tokens=True)["input_ids"])
    best = (abs(full_n - target), text, full_n, len(text))
    while lo <= hi:
        mid = (lo + hi) // 2
        chunk = text[:mid].rstrip()
        n = len(tok(chunk, add_special_tokens=True)["input_ids"])
        if n <= cap:
            if abs(n - target) < best[0]:
                best = (abs(n - target), chunk, n, len(chunk))
            lo = mid + 1
        else:
            hi = mid - 1
    # BPE merges make token count non-monotone in characters; look next to the
    # best cut for an exact hit before settling for the closest.
    _, chunk, n, cut = best
    if n != target:
        for d in range(1, 80):
            for c in (cut - d, cut + d):
                if 1 <= c <= len(text):
                    cand = text[:c].rstrip()
                    cn = len(tok(cand, add_special_tokens=True)["input_ids"])
                    if cn == target and cn <= cap:
                        return cand, cn
    return chunk, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--lengths", default="16,128,512")
    ap.add_argument("--per-length", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=510,
                    help="cap including special tokens (BERT models train at 512)")
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--stream-words", type=int, default=4000,
                    help="length of the corpus stream windows are cut from; a "
                         "2 000-token length needs more than the 4 000-word default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mixed", action="store_true",
                    help="cycle lengths across texts instead of a fixed block per length")
    ap.add_argument("--mixed-set", type=int, default=0,
                    help="also emit this many texts of lengths drawn from 8..512, under the "
                         "\"mixed\" key (S4.9b: the mixed-length cell)")
    ap.add_argument("--mixed-seed", type=int, default=20261111)
    ap.add_argument("--prefix", default="",
                    help="prepend this text to every calibrated text (the retrieval "
                         "instructions some embedders train with, e.g. 'passage: '); the "
                         "token count includes the prefix")
    ap.add_argument("--mixed-prefixes", default="",
                    help="comma-separated prefixes cycled over the mixed set only; overrides "
                         "--prefix for that set (e.g. 'query: ,passage: ')")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x]
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)

    def prefix_cost(pfx):
        """Tokens a prefix really adds to a text (special tokens counted once)."""
        if not pfx:
            return 0
        one = len(tok("a", add_special_tokens=True)["input_ids"])
        return len(tok(pfx + "a", add_special_tokens=True)["input_ids"]) - one

    def calibrate(stream, start, target, prefix):
        """build_text aims at the requested length with the prefix already included."""
        cap = args.max_tokens - prefix_cost(prefix)
        inner = max(1, target - prefix_cost(prefix))
        body, _ = build_text(tok, stream, start, inner, cap)
        text = prefix + body
        n = len(tok(text, add_special_tokens=True)["input_ids"])
        return text, n

    prefix_tokens = prefix_cost(args.prefix)
    mixed_prefixes = [p for p in args.mixed_prefixes.split(",") if p]
    if not mixed_prefixes:
        mixed_prefixes = [args.prefix] if args.prefix else [""]
    words = load_words(args.corpus)
    if len(words) < 300:
        raise SystemExit(f"corpus too short: {len(words)} words")
    # A long stream of the same public corpus, so 512-token texts still have room
    # to start at 128 different offsets; every window is then a distinct text.
    reps = max(1, args.stream_words // len(words) + 1)
    stream = (words * reps)[:args.stream_words]
    span = max(1, len(stream) - 600)

    out = {
        "tokenizer": str(Path(args.tokenizer_dir).resolve()),
        "per_length": args.per_length,
        "max_tokens": args.max_tokens,
        "prefix": args.prefix,
        "lengths": {},
    }
    bad = 0
    for L in lengths:
        texts = []
        for i in range(args.per_length):
            start = (i * 31 + L * 7) % span
            text, n = calibrate(stream, start, L, args.prefix)
            if not (0.95 * L <= n <= 1.05 * L):
                bad += 1
                print(f"[warn] L={L} text {i}: {n} tokens outside 5%", file=sys.stderr)
            texts.append({"text": text, "tokens": n})
        # Distinctness check: identical texts would let a cache hide the work.
        uniq = len({t["text"] for t in texts})
        out["lengths"][str(L)] = texts
        toks = [t["tokens"] for t in texts]
        print(f"L={L}: {uniq}/{len(texts)} distinct, tokens "
              f"{min(toks)}..{max(toks)} (median {sorted(toks)[len(toks)//2]})")
    if args.mixed_set:
        # A fixed seed and a fixed length pool: the same mixed batch for every engine, with
        # lengths from 8 to 512 tokens (the sum is ~14 000 tokens per 32-text batch).
        rng = random.Random(args.mixed_seed)
        pool = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
        texts = []
        for i in range(args.mixed_set):
            target = rng.choice(pool)
            start = (i * 53 + target * 11) % span
            pfx = mixed_prefixes[i % len(mixed_prefixes)]
            text, n = calibrate(stream, start, target, pfx)
            texts.append({"text": text, "tokens": n})
        out["mixed_seed"] = args.mixed_seed
        out["mixed_prefixes"] = mixed_prefixes
        out["mixed_lengths"] = sorted({t["tokens"] for t in texts})
        out["lengths"]["mixed"] = texts
        toks = [t["tokens"] for t in texts]
        print(f"mixed: {len(texts)} texts, tokens {min(toks)}..{max(toks)} "
              f"(sum {sum(toks)}), seed {args.mixed_seed}")
    if bad:
        print(f"[warn] {bad} texts fell outside the 5% window", file=sys.stderr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
