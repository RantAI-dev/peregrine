#!/usr/bin/env python3
"""Embedding benchmark runner (roadmap S4.2 competitor baselines).

Drives an embedding engine adapter over a fixed grid and writes raw JSON:

  lengths  16, 128, 512 tokens (calibrated per model tokenizer, within 5%)
  batches  1, 8, 32, 128 texts per request
  clients  1, plus 10 concurrent clients at batch 1
  rounds   3 interleaved rounds (one round per invocation, merged by embed_report.py)

Reports texts/s, tokens/s per cell and the p50/p99 latency of a whole request.
Standard library only; engine dependencies live in adapters and venv servers.

  python benchmarks/embed.py --select minilm:llama_cpp:f16 --round 1 \
      --output-dir benchmarks/results/bluefin-embed-2026-11-10 --texts-dir /var/tmp/embed-texts
"""

import argparse
import concurrent.futures as futures
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import compute_file_sha256, get_cpu_info, get_governor
from engines.llama_embed import LlamaCppEmbedAdapter
from engines.tei_embed import TEIAdapter
from engines.fastembed_embed import FastEmbedAdapter
from engines.openvino_embed import OpenVINOAdapter
from engines.peregrine_embed import PeregrineEmbedAdapter

DEFAULT_MODEL_DIR = Path(os.environ.get("EMBED_MODEL_DIR", "~/models/embed")).expanduser()
REGISTRY = Path(__file__).resolve().parent / "embed_models.json"

ENGINES = {
    "llama_cpp": LlamaCppEmbedAdapter,
    "tei": TEIAdapter,
    "fastembed": FastEmbedAdapter,
    "openvino": OpenVINOAdapter,
    "peregrine": PeregrineEmbedAdapter,
}


# --------------------------------------------------------------------------- #
# host record and preflight
# --------------------------------------------------------------------------- #
def meminfo() -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            out[k] = int(v.strip().split()[0])  # kB
    except Exception:
        pass
    return out


def host_state() -> Dict[str, Any]:
    mi = meminfo()
    return {
        "loadavg": list(os.getloadavg()),
        "mem_available_gb": round(mi.get("MemAvailable", 0) / 1024 / 1024, 2),
        "mem_total_gb": round(mi.get("MemTotal", 0) / 1024 / 1024, 2),
        "swap_total_gb": round(mi.get("SwapTotal", 0) / 1024 / 1024, 2),
        "swap_free_gb": round(mi.get("SwapFree", 0) / 1024 / 1024, 2),
    }


def machine_record(threads: int) -> Dict[str, Any]:
    cpu = get_cpu_info()
    return {
        "hostname": platform.node(),
        "os": platform.system(),
        "kernel": platform.release(),
        "platform": platform.platform(),
        "cpu_model": cpu["model"],
        "cpu_features": cpu["features"],
        "cores_logical": cpu["cores_logical"],
        "threads_tested": threads,
        "governor": get_governor(),
        "python": platform.python_version(),
    }


def preflight(min_load: float = 0.5, min_avail_gb: float = 5.0) -> Dict[str, Any]:
    st = host_state()
    ok = st["loadavg"][0] < min_load and st["mem_available_gb"] >= min_avail_gb
    if not ok:
        raise SystemExit(
            "host not quiescent: load {:.2f} (need <{}) memory available {:.2f} GB "
            "(need >={} GB); refusing to time".format(
                st["loadavg"][0], min_load, st["mem_available_gb"], min_avail_gb
            )
        )
    return st


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def load_registry(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_rows(reg: Dict[str, Any], select: Optional[List[str]]) -> List[Tuple[str, Dict, Dict]]:
    """Expand selections like `minilm:llama_cpp:f16`, `minilm:llama_cpp`, `minilm`."""
    out = []
    for model_name, model in reg["models"].items():
        for row in model.get("rows", []):
            row = dict(row)
            row["path"] = str((DEFAULT_MODEL_DIR / row["path"]).resolve())
            if select:
                keep = False
                for sel in select:
                    parts = sel.split(":")
                    if parts[0] != model_name:
                        continue
                    if len(parts) > 1 and parts[1] not in ("*", row["engine"]):
                        continue
                    if len(parts) > 2 and parts[2] not in ("*", row["precision"]):
                        continue
                    keep = True
                if not keep:
                    continue
            out.append((model_name, model, row))
    return out


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def pct(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def timed_embed(adapter, texts: List[str]) -> Tuple[float, int, int]:
    t0 = time.perf_counter()
    reply = adapter.embed(texts)
    wall = time.perf_counter() - t0
    n = len(reply.get("embeddings", []))
    if n != len(texts):
        raise RuntimeError(f"engine returned {n} vectors for {len(texts)} texts")
    return wall, int(reply.get("prompt_tokens", 0)), len(texts)


def run_cell(
    adapter,
    texts: List[str],
    calib_tokens: List[int],
    batch: int,
    clients: int,
    budget: float,
    max_requests: int,
    request_timeout: float,
) -> Dict[str, Any]:
    if len(texts) < batch:
        batch = len(texts)
    lat: List[float] = []
    tokens_calib = 0
    tokens_reported = 0
    n_texts = 0
    waves = 0
    t_start = time.perf_counter()
    pool = futures.ThreadPoolExecutor(max_workers=clients)
    try:
        while True:
            elapsed = time.perf_counter() - t_start
            # At least one wave always runs; later waves only while inside the
            # time budget and request count.  A wave that overruns the budget
            # simply ends the cell, so a slow engine is never asked twice.
            if waves > 0 and elapsed >= budget:
                break
            if waves * clients >= max_requests:
                break
            futs = []
            for c in range(clients):
                idx = (waves * clients + c)
                start = (idx * batch) % max(1, len(texts) - batch + 1) if len(texts) > batch else 0
                chunk = texts[start:start + batch]
                cal = calib_tokens[start:start + batch]
                futs.append((pool.submit(timed_embed, adapter, chunk), sum(cal)))
            for fut, cal in futs:
                wall, reported, n = fut.result(timeout=request_timeout)
                lat.append(wall)
                n_texts += n
                tokens_calib += cal
                tokens_reported += reported
            waves += 1
    finally:
        pool.shutdown(wait=True)
    wall = time.perf_counter() - t_start
    # Tokens/s uses the engine-reported count when it is plausible, otherwise the
    # calibrated count (TEI reports no usage). Both are always recorded.
    use_reported = tokens_reported > 0 and abs(tokens_reported - tokens_calib) <= max(
        5, 0.05 * tokens_calib
    )
    tokens = tokens_reported if use_reported else tokens_calib
    return {
        "texts_per_s": round(n_texts / wall, 3),
        "tokens_per_s": round(tokens / wall, 1),
        "wall_s": round(wall, 4),
        "requests": len(lat),
        "texts": n_texts,
        "tokens_calibrated": tokens_calib,
        "tokens_reported": tokens_reported,
        "tokens_source": "engine" if use_reported else "tokenizer",
        "latency_s": {
            "p50": round(pct(lat, 0.50), 6),
            "p99": round(pct(lat, 0.99), 6),
            "min": round(min(lat), 6),
            "max": round(max(lat), 6),
            "n": len(lat),
        },
    }


def path_fingerprint(path: str) -> Dict[str, Any]:
    """SHA-256 of the weights file, or a manifest hash for a model directory."""
    p = Path(path)
    if p.is_file():
        return {"kind": "file", "sha256": compute_file_sha256(p),
                "size_bytes": p.stat().st_size}
    if not p.is_dir():
        return {"kind": "missing", "sha256": "", "size_bytes": 0}
    files = []
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.stat().st_size > 0:
            files.append({"name": str(f.relative_to(p)), "size": f.stat().st_size,
                          "sha256": compute_file_sha256(f)})
    blob = "\n".join(f"{x['name']}:{x['size']}:{x['sha256']}" for x in files)
    return {"kind": "dir", "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
            "size_bytes": sum(x["size"] for x in files), "files": files}


def cell_name(length: int, batch: int, clients: int) -> str:
    return f"L{length}_b{batch}_c{clients}"


def load_skips(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_bench(args) -> None:
    reg = load_registry(REGISTRY)
    rows = resolve_rows(reg, args.select)
    if not rows:
        raise SystemExit("no rows selected")
    lengths = [int(x) for x in args.lengths.split(",") if x]
    batches = [int(x) for x in args.batches.split(",") if x]
    concurrency = [int(x) for x in args.clients.split(",") if x]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts_dir = Path(args.texts_dir)

    machine = machine_record(args.threads)
    pre = preflight()

    for model_name, model, row in rows:
        texts_file = texts_dir / f"{model_name}.json"
        if not texts_file.exists():
            raise SystemExit(f"missing {texts_file}; run embed_texts.py first")
        tdata = json.loads(texts_file.read_text(encoding="utf-8"))
        fp = path_fingerprint(row["path"])
        sha, size = fp["sha256"], fp["size_bytes"]

        max_cell_s = float(
            row.get("max_cell_seconds") or model.get("max_cell_seconds") or args.max_cell_seconds
        )
        # The skip decision is made once (round 1) and reused by later rounds so
        # that every round covers exactly the same cells.
        skips_path = out_dir / f"{model_name}-{row['engine']}-{row['precision']}-skips.json"
        skipped: Dict[str, Any] = load_skips(skips_path) or {}
        decide_skips = not skips_path.exists()

        # One server per (model, precision): the port is fixed so a crashed
        # server cannot leave a stale process on a different port.
        port = args.port + (hash(f"{model_name}{row['engine']}{row['precision']}") % 50)
        adapter = make_adapter(row["engine"], args)
        version = adapter.get_version_info()
        print(f"\n== {model_name} / {row['engine']} / {row['precision']} "
              f"(round {args.round}) ==", flush=True)
        started = adapter.start(row, model, port)
        try:
            # Warm-up: one request per length, excluded from the timing.  The
            # second single-text request is timed and used to predict the cost
            # of the big-batch cells.
            single_s: Dict[int, float] = {}
            for L in lengths:
                item = tdata["lengths"][str(L)]
                adapter.embed([t["text"] for t in item[:2]])
                t0 = time.perf_counter()
                adapter.embed([item[0]["text"]])
                single_s[L] = time.perf_counter() - t0
            state_before = host_state()
            cells: Dict[str, Any] = {}
            for L in lengths:
                item = tdata["lengths"][str(L)]
                all_texts = [t["text"] for t in item]
                all_tokens = [t["tokens"] for t in item]
                for batch in batches:
                    for clients in concurrency:
                        if clients > 1 and batch != 1:
                            continue
                        key = cell_name(L, batch, clients)
                        # Conservative: one text costs what it cost alone, so the
                        # estimate never under-predicts a serial engine.
                        est = single_s[L] * batch
                        if key in skipped:
                            cells[key] = dict(skipped[key])
                            continue
                        if decide_skips and batch > 1 and est > max_cell_s:
                            reason = (
                                f"estimated {est:.0f} s for {batch} x {L} tokens on this row "
                                f"(single text {single_s[L] * 1000:.0f} ms) exceeds the "
                                f"{max_cell_s:.0f} s per-cell cap; skipped to keep the host "
                                f"inside its memory and wall-clock budget"
                            )
                            skipped[key] = {"skipped": reason, "estimate_s": round(est, 1),
                                            "single_text_s": round(single_s[L], 4)}
                            cells[key] = dict(skipped[key])
                            print(f"  {key:20s} SKIP ({est:.0f}s estimate)", flush=True)
                            continue
                        budget = args.budget * (3.0 if clients > 1 else 1.0)
                        max_reqs = (args.max_requests if batch == 1 else args.max_big_requests)
                        if clients > 1:
                            max_reqs = max(clients * 6, clients)
                        elif batch > 1 and single_s[L] * batch > args.budget:
                            max_reqs = 1
                        t0 = time.perf_counter()
                        res = run_cell(
                            adapter, all_texts, all_tokens, batch, clients,
                            budget=budget, max_requests=max_reqs,
                            request_timeout=args.request_timeout,
                        )
                        res.update({
                            "length": L,
                            "batch": batch,
                            "clients": clients,
                            "elapsed_s": round(time.perf_counter() - t0, 2),
                            "tokens_per_text_median": int(statistics.median(all_tokens)),
                        })
                        cells[key] = res
                        print(f"  {key:20s} {res['texts_per_s']:8.2f} texts/s "
                              f"{res['tokens_per_s']:9.1f} tok/s "
                              f"p50 {res['latency_s']['p50'] * 1000:7.1f} ms "
                              f"p99 {res['latency_s']['p99'] * 1000:8.1f} ms "
                              f"n={res['requests']} ({res['elapsed_s']}s)", flush=True)
            state_after = host_state()
        finally:
            adapter.stop()

        if decide_skips:
            skips_path.write_text(json.dumps(skipped, indent=1), encoding="utf-8")

        record = {
            "benchmark": "peregrine-s4.2-embed",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "round": args.round,
            "model": {
                "name": model_name,
                "hf": model.get("hf"),
                "pooling": model.get("pooling"),
                "max_tokens": model.get("max_tokens"),
                "dim": model.get("dim"),
                "tokenizer_dir": model.get("tokenizer_dir"),
            },
            "row": {
                "engine": row["engine"],
                "precision": row["precision"],
                "path": row["path"],
                "sha256": sha,
                "size_bytes": size,
                "fingerprint": fp,
                "notes": row.get("notes", ""),
            },
            "machine": machine,
            "engine": version,
            "start": started,
            "host_before": state_before,
            "host_after": state_after,
            "preflight": pre,
            "settings": {
                "lengths": lengths,
                "batches": batches,
                "clients": concurrency,
                "threads": args.threads,
                "budget_s": args.budget,
                "max_requests": args.max_requests,
                "max_big_requests": args.max_big_requests,
                "max_cell_seconds": max_cell_s,
                "texts_file": str(texts_file),
                "texts_per_length": tdata["per_length"],
            },
            "cells": cells,
        }
        fname = f"{model_name}-{row['engine']}-{row['precision']}-r{args.round}.json"
        (out_dir / fname).write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"  wrote {out_dir / fname}", flush=True)


# --------------------------------------------------------------------------- #
# accuracy
# --------------------------------------------------------------------------- #
def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def run_accuracy(args) -> None:
    reg = load_registry(REGISTRY)
    rows = resolve_rows(reg, args.select)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts_dir = Path(args.texts_dir)
    # Accuracy runs are not timed, so only the memory half of the preflight
    # applies; the load average is still recorded.
    st = host_state()
    if st["mem_available_gb"] < 5.0:
        raise SystemExit(
            f"only {st['mem_available_gb']:.2f} GB available; refusing to load a model"
        )
    pre = st

    for model_name, model, row in rows:
        ref_file = texts_dir / "ref" / f"{model_name}-{args.ref_label}.json"
        if not ref_file.exists():
            print(f"[skip] no reference {ref_file} for {model_name}", flush=True)
            continue
        ref = json.loads(ref_file.read_text(encoding="utf-8"))
        tdata = json.loads((texts_dir / f"{model_name}-acc.json").read_text(encoding="utf-8"))
        texts: List[str] = []
        for L in ref["lengths"]:
            texts.extend(t["text"] for t in tdata["lengths"][str(L)])
        texts = texts[: len(ref["vectors"])]

        adapter = make_adapter(row["engine"], args)
        print(f"\n== accuracy {model_name} / {row['engine']} / {row['precision']} ==", flush=True)
        adapter.start(row, model, args.port)
        try:
            got: List[List[float]] = []
            for i in range(0, len(texts), args.acc_batch):
                got.extend(adapter.embed(texts[i:i + args.acc_batch])["embeddings"])
        finally:
            adapter.stop()

        cos = [cosine(g, r) for g, r in zip(got, ref["vectors"])]
        norms = [math.sqrt(sum(x * x for x in g)) for g in got]
        record = {
            "benchmark": "peregrine-s4.2-embed-accuracy",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": {"name": model_name, "hf": model.get("hf"), "pooling": model.get("pooling")},
            "row": {"engine": row["engine"], "precision": row["precision"],
                    "path": row["path"], **{k: v for k, v in path_fingerprint(row["path"]).items()
                                            if k != "files"}},
            "reference": {"label": args.ref_label, "backend": ref.get("backend"),
                          "dim": ref.get("dim"), "count": len(ref["vectors"])},
            "engine": adapter.get_version_info(),
            "host_before": pre,
            "cosine": {
                "mean": round(statistics.fmean(cos), 6),
                "min": round(min(cos), 6),
                "p1": round(pct(cos, 0.01), 6),
                "p50": round(pct(cos, 0.50), 6),
            },
            "norm": {
                "min": round(min(norms), 6),
                "max": round(max(norms), 6),
                "mean": round(statistics.fmean(norms), 6),
            },
            "dim": len(got[0]) if got else 0,
            "n": len(cos),
        }
        fname = f"accuracy-{model_name}-{row['engine']}-{row['precision']}.json"
        (out_dir / fname).write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"  cosine mean {record['cosine']['mean']:.6f} min {record['cosine']['min']:.6f}; "
              f"norm {record['norm']['mean']:.4f}; dim {record['dim']}; wrote {fname}", flush=True)


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Embedding benchmark (S4.2)")
    ap.add_argument("--select", action="append",
                    help="model[:engine[:precision]], repeatable; default all rows")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--lengths", default="16,128,512")
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--clients", default="1,10")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--budget", type=float, default=2.0, help="seconds of timed requests per cell")
    ap.add_argument("--max-requests", type=int, default=60)
    ap.add_argument("--max-big-requests", type=int, default=3)
    ap.add_argument("--max-cell-seconds", type=float, default=120.0,
                    help="skip a batch cell whose predicted single-request time exceeds this")
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--port", type=int, default=8401)
    ap.add_argument("--texts-dir", default="/var/tmp/embed-texts")
    ap.add_argument("--output-dir", default="benchmarks/results/bluefin-embed-2026-11-10")
    ap.add_argument("--mode", choices=["bench", "accuracy"], default="bench")
    ap.add_argument("--ref-label", default="torch-fp32")
    ap.add_argument("--acc-batch", type=int, default=16)
    ap.add_argument("--peregrine-worktree", default=None,
                    help="worktree whose target/release/peregrine serves /v1/embeddings (S4.1)")
    ap.add_argument("--peregrine-binary", default=None)
    return ap.parse_args()


def make_adapter(engine: str, args):
    cls = ENGINES[engine]
    if engine == "peregrine":
        return cls(threads=args.threads, binary=args.peregrine_binary,
                   worktree=args.peregrine_worktree)
    return cls(threads=args.threads)


def main():
    args = parse_args()
    if args.mode == "bench":
        run_bench(args)
    else:
        run_accuracy(args)


if __name__ == "__main__":
    main()
