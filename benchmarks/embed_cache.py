#!/usr/bin/env python3
"""Repeated-text embedding benchmark: the exact result cache on and off (S4.9k).

Starts one `peregrine serve` per cache setting and measures two shapes a cache
can serve: the same text requested again and again (single stream), and a batch
in which every text is the same. The same text is unique work for the engine on
every request when the cache is off, so the pair is the change's before/after.

  python benchmarks/embed_cache.py --binary target/release/peregrine \
      --model ~/models/embed/embeddinggemma/embeddinggemma-Q8_0.gguf \
      --texts /var/tmp/embed-texts-s49k/embeddinggemma.json
"""

import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time
import urllib.request


def post(url, texts, timeout=600.0):
    body = json.dumps({"input": texts, "model": "embed"}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


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
    ap.add_argument("--lengths", default="16,128,512")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--single-requests", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--cache-mb", default="0,64")
    ap.add_argument("--port", type=int, default=8391)
    args = ap.parse_args()

    tdata = json.loads(args.texts.read_text(encoding="utf-8"))
    lengths = [int(x) for x in args.lengths.split(",") if x]
    results = {}
    for mb in [int(x) for x in args.cache_mb.split(",") if x]:
        port = free_port(args.port)
        log_path = Path("/var/tmp") / f"embed-cache-{port}.log"
        log = open(log_path, "wb")
        env = dict(os.environ)
        env["PEREGRINE_EMBED_CACHE_MB"] = str(mb)
        proc = subprocess.Popen(
            [str(Path(args.binary).expanduser()), "serve", "-m",
             str(Path(args.model).expanduser()), "--host", "127.0.0.1",
             "--port", str(port), "-t", str(args.threads), "-c", str(args.ctx),
             "--slots", "2", "--no-profile"],
            stdout=log, stderr=subprocess.STDOUT, env=env,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            if not wait_ready(base + "/v1/models", proc):
                raise SystemExit(f"server failed: {log_path}")
            post(base + "/v1/embeddings", [tdata["lengths"]["16"][0]["text"]])  # warm-up
            for L in lengths:
                text = tdata["lengths"][str(L)][0]["text"]
                lat = []
                for _ in range(args.single_requests):
                    t0 = time.perf_counter()
                    post(base + "/v1/embeddings", [text])
                    lat.append(time.perf_counter() - t0)
                batch = []
                for _ in range(args.rounds):
                    t0 = time.perf_counter()
                    post(base + "/v1/embeddings", [text] * args.batch)
                    batch.append(time.perf_counter() - t0)
                results.setdefault(mb, {})[L] = {
                    "single_ms": round(statistics.median(lat) * 1e3, 3),
                    "batch_ms": round(statistics.median(batch) * 1e3, 3),
                    "batch_texts_per_s": round(args.batch / statistics.median(batch), 2),
                }
                print(f"cache {mb:3d} MB  L{L:<5d} single {results[mb][L]['single_ms']:8.2f} ms  "
                      f"{args.batch} identical {results[mb][L]['batch_ms']:8.1f} ms "
                      f"({results[mb][L]['batch_texts_per_s']:.1f} texts/s)", flush=True)
            metrics = urllib.request.urlopen(base + "/metrics").read().decode()
            for line in metrics.splitlines():
                if "embed_cache" in line:
                    print("  " + line)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:
                proc.kill()
            log.close()

    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
