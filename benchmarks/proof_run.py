#!/usr/bin/env python3
"""Drive the S4.9l proof run: every embedding row, one invocation at a time.

`benchmarks/embed.py` measures one row (model, engine, precision) per invocation,
which is what keeps one model process on the host and every command short. This
driver holds the proof run's row list and its settings, runs the missing rows in
order with an interleaved round, and stops before `--max-seconds` so the caller
can start the next chunk. Re-running it continues where the output directory
left off (a row whose JSON exists is skipped).

  python benchmarks/proof_run.py --mode bench --round 1 \
      --texts-dir /var/tmp/embed-texts-s49l \
      --out-dir benchmarks/results/bluefin-2026-11-12-s4.9l-proof/raw

Modes:

  bench     the timing grid, exactly as embed.py records it; Peregrine runs with
            `PEREGRINE_EMBED_CACHE_MB=0` because every round repeats the same texts
            and a cache hit is not a computation
  cache     the same Peregrine rows with the default 64 MiB exact cache on: the
            "repeated traffic" table, written to its own directory
  accuracy  `embed.py --mode accuracy` for the rows that have a reference JSON

Every row measures batches 1, 8 and 32 at 16/128/512 tokens (2 000 where the
model allows it) plus the fixed-seed mixed-length batch, one client, a 0.5 s
timed budget per cell, medians of three rounds taken later by the report.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EMBED = HERE / "embed.py"

# llama.cpp needs the whole non-causal prompt in one ubatch, so a row with
# 2 000-token texts runs `-b 2048 -ub 2048`; the 512-token rows keep 1 024.
LLAMA_1024 = ["--llama-batch", "1024", "--llama-ubatch", "1024"]
LLAMA_2048 = ["--llama-batch", "2048", "--llama-ubatch", "2048"]

SHORT = [16, 128, 512]
LONG = [16, 128, 512, 2000]


def row(model, engine, precision, lengths, extra=(), ctx=0, ref_label="torch-fp32"):
    return {
        "model": model,
        "engine": engine,
        "precision": precision,
        "lengths": lengths,
        "extra": list(extra),
        "ctx": ctx,
        "ref_label": ref_label,
    }


ROWS = [
    # all-MiniLM-L6-v2: the small BERT encoder every Python engine targets.
    row("minilm", "peregrine", "f16", SHORT),
    row("minilm", "peregrine", "q8_0", SHORT),
    row("minilm", "llama_cpp", "f16", SHORT, LLAMA_1024),
    row("minilm", "llama_cpp", "q8_0", SHORT, LLAMA_1024),
    row("minilm", "tei", "fp32", SHORT),
    row("minilm", "fastembed", "fp32", SHORT),
    row("minilm", "fastembed", "int8", SHORT),
    row("minilm", "openvino", "fp32", SHORT),
    row("minilm", "onnxruntime", "fp32", SHORT),
    row("minilm", "onnxruntime", "int8", SHORT),
    # bge-small-en-v1.5: the same shape, CLS pooling.
    row("bge-small", "peregrine", "f32", SHORT),
    row("bge-small", "peregrine", "q8_0", SHORT),
    row("bge-small", "llama_cpp", "f32", SHORT, LLAMA_1024),
    row("bge-small", "llama_cpp", "q8_0", SHORT, LLAMA_1024),
    row("bge-small", "tei", "fp32", SHORT),
    row("bge-small", "fastembed", "fp32", SHORT),
    row("bge-small", "fastembed", "int8", SHORT),
    row("bge-small", "openvino", "fp32", SHORT),
    row("bge-small", "onnxruntime", "fp32", SHORT),
    row("bge-small", "onnxruntime", "int8", SHORT),
    # bge-m3 (XLM-R): the registry row carries llama.cpp's -np 8 / -c 16448.
    row("bge-m3", "peregrine", "q8_0", LONG, ctx=2048),
    row("bge-m3", "llama_cpp", "q8_0", LONG),
    # EmbeddingGemma 300M and Qwen3-Embedding 0.6B: llama.cpp and Peregrine only.
    row("embeddinggemma", "peregrine", "q8_0", LONG, ctx=2048),
    row("embeddinggemma", "llama_cpp", "q8_0", LONG, LLAMA_2048),
    row("qwen3-embed-0.6b", "peregrine", "f16", LONG, ctx=2048),
    row("qwen3-embed-0.6b", "peregrine", "q8_0", LONG, ctx=2048),
    row("qwen3-embed-0.6b", "llama_cpp", "f16", LONG,
        LLAMA_2048 + ["--llama-parallel", "10", "--llama-ctx", "20560"], ctx=2048),
    row("qwen3-embed-0.6b", "llama_cpp", "q8_0", LONG,
        LLAMA_2048 + ["--llama-parallel", "10", "--llama-ctx", "20560"], ctx=2048),
    # Nemotron 3.5 Embed 1B (mistral3 encoder): the S4.14b path.
    row("nemotron-3.5-embed-1b", "peregrine", "q8_0", LONG, ctx=2048),
    row("nemotron-3.5-embed-1b", "llama_cpp", "q8_0", LONG, LLAMA_2048),
]

ACCURACY_REF_LABEL = {
    "embeddinggemma": "onnx-fp32",
}


def out_name(r, mode, round_no):
    if mode == "accuracy":
        return f"accuracy-{r['model']}-{r['engine']}-{r['precision']}.json"
    return f"{r['model']}-{r['engine']}-{r['precision']}-r{round_no}.json"


def command(r, args, mode):
    cmd = [sys.executable, str(EMBED),
           "--select", f"{r['model']}:{r['engine']}:{r['precision']}",
           "--round", str(args.round),
           "--threads", "4",
           "--max-load", str(args.max_load),
           "--min-avail-gb", str(args.min_avail_gb),
           "--port", str(args.port),
           "--texts-dir", str(args.texts_dir),
           "--output-dir", str(args.out_dir)]
    if args.peregrine_binary:
        cmd += ["--peregrine-binary", args.peregrine_binary]
    if mode == "accuracy":
        cmd += ["--mode", "accuracy", "--acc-batch", "16",
                "--ref-label", r["ref_label"]]
        return cmd
    cmd += ["--lengths", ",".join(str(x) for x in r["lengths"]),
            "--batches", "1,8,32",
            "--clients", "1",
            "--budget", str(args.budget),
            "--max-requests", "20",
            "--max-big-requests", "2",
            "--max-cell-seconds", str(args.max_cell_seconds),
            "--request-timeout", str(args.request_timeout),
            "--mixed-cell", "--mixed-batch", "32"]
    if r["ctx"]:
        cmd += ["--ctx-tokens", str(r["ctx"])]
    if mode == "cache":
        cmd.append("--repeat-texts")
    cmd += r["extra"]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "cache", "accuracy"], default="bench")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--texts-dir", type=Path, default=Path("/var/tmp/embed-texts-s49l"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--peregrine-binary", default=None)
    ap.add_argument("--models", default="",
                    help="comma-separated model names; default every row")
    ap.add_argument("--engines", default="",
                    help="comma-separated engines; default every row")
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--max-cell-seconds", type=float, default=900.0)
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument("--max-load", type=float, default=20.0,
                    help="the host is shared; the load is recorded, and the gate only "
                         "refuses a host that is actually busy")
    ap.add_argument("--min-avail-gb", type=float, default=5.0,
                    help="refuse to time below this much free memory; the small encoder "
                         "rows need a fraction of it, so a shared desktop can lower it")
    ap.add_argument("--port", type=int, default=8401)
    ap.add_argument("--max-seconds", type=float, default=480.0,
                    help="stop and report between rows after this much wall time")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = {m for m in args.models.split(",") if m}
    engines = {e for e in args.engines.split(",") if e}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in ROWS
            if (not models or r["model"] in models)
            and (not engines or r["engine"] in engines)
            and (args.mode != "cache" or r["engine"] == "peregrine")]
    if args.mode == "accuracy":
        rows = [r for r in rows if r["model"] in
                ("minilm", "bge-small", "bge-m3", "embeddinggemma", "qwen3-embed-0.6b")]
        for r in rows:
            r["ref_label"] = ACCURACY_REF_LABEL.get(r["model"], r["ref_label"])

    env = dict(os.environ)
    if args.mode == "bench":
        env["PEREGRINE_EMBED_CACHE_MB"] = "0"
    elif args.mode == "cache":
        env.pop("PEREGRINE_EMBED_CACHE_MB", None)  # the server default, 64 MiB

    t0 = time.time()
    ran = 0
    for r in rows:
        name = out_name(r, args.mode, args.round)
        if (args.out_dir / name).exists():
            continue
        cmd = command(r, args, args.mode)
        print(f"\n### {r['model']} {r['engine']} {r['precision']} ({args.mode}, "
              f"round {args.round})", flush=True)
        if args.dry_run:
            print(" ".join(cmd))
            ran += 1
            continue
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            print(f"[fail] exit {res.returncode}: {' '.join(cmd)}", file=sys.stderr)
            sys.exit(res.returncode)
        ran += 1
        if time.time() - t0 > args.max_seconds:
            print(f"[chunk] {ran} rows this invocation; stopping before the next row",
                  flush=True)
            break
    print(f"[done] {ran} rows in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
