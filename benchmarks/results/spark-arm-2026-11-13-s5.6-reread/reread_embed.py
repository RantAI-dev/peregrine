#!/usr/bin/env python3
"""The S5.6 re-read's embedding cell: three builds interleaved in every round.

`../spark-arm-2026-11-13-s5.6/embed_wave.py` measures one engine. This starts one
`peregrine serve` per arm on its own port, warms each with one unmeasured wave, then
for every round and disturbance level sends the 32 different 128-token texts
concurrently to each arm in turn and times the wave. The exact result cache is off
(`--opt EMBED_CACHE_MB=0`); the disturbance generator is started for one (level,
round) measurement and stopped by PID in a `finally`.

  python reread_embed.py --arms head,merged,gap0 --threads 10 --rounds 0,1,2 \
      --out raw/reread-embed-r012.jsonl
"""

import argparse
import json
import os
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
HOME = os.path.expanduser("~")
MODEL = f"{HOME}/models/gte-modernbert-base-F16.gguf"
TEXTS = "/tmp/opencode/texts-128x32.json"
CLUSTER = "5-9,15-19"
IDLE_CPUS = (5, 6, 7, 8, 9, 15, 16, 17, 18, 19)
ARMS = {
    "head": (f"{HOME}/wt-s56-head/target/release/peregrine", 18310),
    "merged": (f"{HOME}/wt-s56-reread/target/release/peregrine", 18320),
    "gap0": (f"{HOME}/wt-s56-gap0/target/release/peregrine", 18330),
    "fixed": (f"{HOME}/peregrine/target/release/peregrine", 18340),
}
LOADGEN = f"{HOME}/wt-s56-reread/target/release/examples/loadgen"


def idle_snapshot():
    out = {}
    for line in open("/proc/stat"):
        parts = line.split()
        if parts and parts[0].startswith("cpu") and parts[0] != "cpu":
            cpu = int(parts[0][3:])
            if cpu in IDLE_CPUS:
                vals = [int(v) for v in parts[1:]]
                out[cpu] = (vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals[:8]))
    return out


def idle_share(a, b):
    vals = []
    for c in a:
        di, dt = b[c][0] - a[c][0], b[c][1] - a[c][1]
        if dt > 0:
            vals.append(di / dt)
    return sum(vals) / len(vals) if vals else 0.0


def load1():
    return float(open("/proc/loadavg").read().split()[0])


def texts():
    d = json.load(open(TEXTS))
    return [t["text"] for t in d["128"]]


def start(arm, threads):
    binp, port = ARMS[arm]
    env = dict(os.environ)
    env["PEREGRINE_EMBED_CACHE_MB"] = "0"
    argv = [binp, "serve", "-m", MODEL, "--threads", str(threads), "--port", str(port),
            "--slots", "32", "--opt", "EMBED_CACHE_MB=0", "--opt", "PACK_CACHE=0",
            "--no-profile"]
    p = subprocess.Popen(["taskset", "-c", CLUSTER] + argv, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 180
    while time.time() < deadline:
        if p.poll() is not None:
            raise RuntimeError(f"{arm} server exited {p.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=1) as r:
                if r.status == 200:
                    return p, base
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"{arm} server did not become ready")


def wave(base, body_texts, timeout):
    out = [None] * len(body_texts)
    err = []

    def one(i):
        req = urllib.request.Request(
            base + "/v1/embeddings",
            data=json.dumps({"input": body_texts[i], "model": "x"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode())
                out[i] = len(d["data"][0]["embedding"])
        except Exception as e:  # noqa: BLE001
            err.append(str(e)[:200])

    ts = [threading.Thread(target=one, args=(i,)) for i in range(len(body_texts))]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return time.time() - t0, out, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="head,merged,gap0")
    ap.add_argument("--threads", default="10")
    ap.add_argument("--levels", default="0,1,2,4")
    ap.add_argument("--rounds", default="0,1,2,3,4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    body = texts()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    levels = [int(x) for x in args.levels.split(",") if x != ""]
    threads = int(args.threads)
    servers = {}
    try:
        for arm in arms:
            servers[arm] = start(arm, threads)
        for arm in arms:
            wave(servers[arm][1], body, args.timeout)  # warm-up, not measured
        for r in [int(x) for x in args.rounds.split(",") if x != ""]:
            for level in levels:
                dist = None
                try:
                    if level > 0:
                        dist = subprocess.Popen(
                            ["taskset", "-c", CLUSTER, LOADGEN, str(level)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        time.sleep(1.0)
                    for arm in arms:
                        l1 = load1()
                        before = idle_snapshot()
                        wall, dims, err = wave(servers[arm][1], body, args.timeout)
                        idle = idle_share(before, idle_snapshot())
                        rec = {"arm": arm, "level": level, "threads": threads,
                               "cell": "embed32x128", "run": r,
                               "value": round(len(body) / wall, 3) if not err else 0.0,
                               "wall_s": round(wall, 3), "idle": round(idle, 3),
                               "load1": l1, "errors": err[:3]}
                        with open(out, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                            f.flush()
                        print(f'r{r} {arm:6} t{threads} l{level} embed {rec["value"]:8.2f} '
                              f'texts/s load {rec["load1"]:5.2f} idle {rec["idle"]:.2f} '
                              f'{rec["errors"][:1]}', flush=True)
                finally:
                    if dist is not None:
                        dist.send_signal(signal.SIGTERM)
                        try:
                            dist.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            dist.kill()
                            dist.wait(timeout=5)
    finally:
        for arm, (p, _) in servers.items():
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)


if __name__ == "__main__":
    main()
