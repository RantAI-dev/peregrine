#!/usr/bin/env python3
"""Quiet-host A/B of two Peregrine binaries on the 32×128 embedding cell.

Two servers, one per binary, up at once on the same ten CPUs; the waves alternate
so host drift hits both alike. This is the control that shows the stall machinery
costs nothing when the host is quiet (the grid's own quiet wave drifts by 15 %).

  python embed_quiet_ab.py <before-binary> <after-binary> [rounds]
"""

import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HOME = os.path.expanduser("~")
MODEL = f"{HOME}/models/gte-modernbert-base-F16.gguf"
TEXTS = "/tmp/opencode/texts-128x32.json"
CLUSTER = "5-9,15-19"


def start(binary, port):
    env = {**os.environ, "PEREGRINE_EMBED_CACHE_MB": "0"}
    p = subprocess.Popen(
        ["taskset", "-c", CLUSTER, binary, "serve", "-m", MODEL, "--threads", "10",
         "--port", str(port), "--slots", "32", "--opt", "EMBED_CACHE_MB=0",
         "--opt", "PACK_CACHE=0", "--no-profile"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(240):
        if p.poll() is not None:
            raise SystemExit(f"{binary} exited {p.returncode}")
        try:
            urllib.request.urlopen(base + "/health", timeout=1).read()
            return p, base
        except Exception:
            time.sleep(0.5)
    raise SystemExit(f"{binary} did not become ready")


def wave(base, body):
    def one(text):
        req = urllib.request.Request(
            base + "/v1/embeddings",
            data=json.dumps({"input": text, "model": "peregrine"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=120).read()

    ts = [threading.Thread(target=one, args=(t,)) for t in body]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return len(body) / (time.time() - t0)


def main():
    before, after = sys.argv[1], sys.argv[2]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    body = [t["text"] for t in json.load(open(TEXTS))["128"]]
    a, ba = start(before, 18401)
    b, bb = start(after, 18402)
    try:
        wave(ba, body)
        wave(bb, body)
        A, B = [], []
        for _ in range(rounds):
            A.append(wave(ba, body))
            B.append(wave(bb, body))
        print("before", [round(x, 2) for x in A], "median", round(statistics.median(A), 2))
        print("after ", [round(x, 2) for x in B], "median", round(statistics.median(B), 2))
    finally:
        for p in (a, b):
            p.send_signal(signal.SIGTERM)
        for p in (a, b):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
