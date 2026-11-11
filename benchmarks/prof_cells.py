#!/usr/bin/env python3
"""Per-cell stage profile of Peregrine's embedding engine (`PEREGRINE_PROF2`).

The throughput harness runs with `--no-profile`; this script runs the same cells
for a few seconds with the phase timers on, reads the server's `prof:` lines
from stderr, and stores per-cell stage shares: matmul (by projection), attention,
quantise, barrier wait and the rest.  The request wall is timed too, so the
cell's `http_tokenise_residual_ms` is the part of a request the engine's own
counters do not cover (HTTP, tokenisation, admission and queueing).

  python benchmarks/prof_cells.py --binary target/release/peregrine \
      --model ~/models/embed/embeddinggemma/embeddinggemma-Q8_0.gguf \
      --texts /var/tmp/embed-texts-s49k/embeddinggemma.json \
      --output-dir .../profile
"""

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

PROF = re.compile(r"^prof: (\d+) steps, (\d+) rows, wall ([\d.]+) ms, per step ([\d.]+) ms, "
                  r"rows/step ([\d.]+); (.*)$")


def post(url, body, timeout=600.0):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_ready(url, proc, timeout=180.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def free_port(start):
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="target/release/peregrine")
    ap.add_argument("--model", required=True)
    ap.add_argument("--texts", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--lengths", default="16,128,512,2000")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--mixed-batch", type=int, default=32)
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--seconds", type=float, default=2.5,
                    help="how long each cell runs under the profiler")
    ap.add_argument("--port", type=int, default=8371)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tdata = json.loads(args.texts.read_text(encoding="utf-8"))
    port = free_port(args.port)
    log_path = Path("/var/tmp") / f"prof-cells-{port}.log"
    log = open(log_path, "wb")
    env = dict(os.environ)
    env["PEREGRINE_PROF2"] = "1"
    cmd = [str(Path(args.binary).expanduser()), "serve", "-m", str(Path(args.model).expanduser()),
           "--host", "127.0.0.1", "--port", str(port), "-t", str(args.threads),
           "-c", str(args.ctx), "--slots", "10", "--no-profile"]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    base = f"http://127.0.0.1:{port}"
    if not wait_ready(base + "/v1/models", proc):
        proc.kill()
        raise SystemExit(f"server failed:\n{log_path.read_text(errors='replace')[-3000:]}")

    def read_prof():
        log.flush()
        lines = []
        with open(log_path, "r", errors="replace") as f:
            for ln in f:
                m = PROF.match(ln.strip())
                if m:
                    lines.append(m.groups())
        return lines

    cells = {}
    try:
        def run_cell(key, texts, batch):
            # Warm-up (not profiled): template, allocations and first weight pass.
            post(base + "/v1/embeddings", {"input": texts[:batch], "model": "embed"})
            start = time.perf_counter()
            reqs = 0
            prof_before = len(read_prof())
            while time.perf_counter() - start < args.seconds:
                t0 = time.perf_counter()
                post(base + "/v1/embeddings", {"input": texts[:batch], "model": "embed"})
                reqs += 1
                last = time.perf_counter() - t0
            wall = time.perf_counter() - start
            lines = read_prof()[prof_before:]
            tot = {}
            steps = rows = 0
            engine_ms = 0.0
            for nsteps, nrows, w, per_step, rps, parts in lines:
                steps += int(nsteps)
                rows += int(nrows)
                engine_ms += float(w)
                for part in parts.split(", "):
                    m = re.match(r"^(.*) ([\d.]+)$", part)
                    if not m:
                        continue
                    name, ms = m.group(1).strip(), float(m.group(2))
                    tot[name] = tot.get(name, 0.0) + ms
            matmul = sum(v for k, v in tot.items() if k.startswith("matmul") or k == "output head")
            buckets = {
                "matmul": matmul,
                "attention": tot.get("attention", 0.0),
                "quantise": tot.get("quantise", 0.0),
                "barrier_wait": tot.get("barrier wait", 0.0),
                "norms": tot.get("norms", 0.0),
                "rope": tot.get("rope+bias", 0.0),
                "kv_write": tot.get("kv write", 0.0),
                "residual": tot.get("residual", 0.0),
                "swiglu": tot.get("swiglu", 0.0),
                "other": tot.get("other", 0.0),
            }
            denom = sum(buckets.values()) or 1.0
            request_wall_ms = wall * 1e3 / max(1, reqs)
            rows_per_req = rows / max(1, reqs)
            cells[key] = {
                "batch": batch,
                "requests": reqs,
                "wall_s": round(wall, 3),
                "request_wall_ms": round(request_wall_ms, 3),
                "engine_wall_ms_per_request": round(engine_ms / max(1, reqs), 3),
                "engine_wall_ms_per_row": round(engine_ms / max(1, rows), 4),
                "rows_per_request": round(rows_per_req, 1),
                "prof_lines": len(lines),
                "stages_ms": {k: round(v, 2) for k, v in buckets.items()},
                "stages_share": {k: round(100.0 * v / denom, 2) for k, v in buckets.items()},
                "http_tokenise_residual_ms": round(
                    request_wall_ms - engine_ms / max(1, reqs), 3),
                "last_request_ms": round(last * 1e3, 3),
            }
            print(f"  {key:14s} req {request_wall_ms:8.2f} ms  engine {engine_ms / max(1, reqs):8.2f} ms  "
                  f"mm {100 * buckets['matmul'] / denom:5.1f}%  attn "
                  f"{100 * buckets['attention'] / denom:5.1f}%  q "
                  f"{100 * buckets['quantise'] / denom:4.1f}%  wait "
                  f"{100 * buckets['barrier_wait'] / denom:4.1f}%  (lines {len(lines)})",
                  flush=True)

        lengths = [int(x) for x in args.lengths.split(",") if x]
        for L in lengths:
            item = tdata["lengths"][str(L)]
            texts = [t["text"] for t in item]
            for b in [int(x) for x in args.batches.split(",") if x]:
                if b * L > 32768:
                    continue
                run_cell(f"L{L}_b{b}", texts, b)
        if args.mixed and "mixed" in tdata["lengths"]:
            texts = [t["text"] for t in tdata["lengths"]["mixed"]]
            run_cell(f"mixed_b{args.mixed_batch}", texts, args.mixed_batch)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        log.close()

    out = {
        "model": str(Path(args.model).expanduser()),
        "threads": args.threads,
        "texts": str(args.texts),
        "method": "PEREGRINE_PROF2=1 server prof lines, summed over the cell; "
                  "each cell ran for --seconds after one warm-up request",
        "cells": cells,
    }
    name = Path(args.model).stem + f"-t{args.threads}-profile.json"
    (args.output_dir / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.output_dir / name}")


if __name__ == "__main__":
    main()
