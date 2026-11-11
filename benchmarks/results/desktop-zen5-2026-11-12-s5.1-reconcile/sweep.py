#!/usr/bin/env python3
"""Round 3's end-to-end driver (S5.1, desktop Zen 5, 2026-11-12).

One harness for both engines, one point at a time, points interleaved round by round so a
slow patch of the host lands on every column. Every row keeps the raw sample times *and*
the child's CPU time, because the two clocks this host exposes disagree:

  * `CLOCK_MONOTONIC` (what both engines and this script time with) is what a user waits
    for, and is the quantity the engines report;
  * `getrusage(RUSAGE_CHILDREN)` CPU time is the scheduler's accounting of the same work.
    For a one-thread run wall and CPU must agree to within a few percent; when they do not,
    the run was descheduled (stolen by the Windows side) or the guest clock moved, and the
    row says so.

    python3 sweep.py <results-dir> <series>

Series: fork-1t, fork-12t, peregrine-1t, peregrine-12t, clock.
"""

import json
import os
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PEREGRINE = os.path.expanduser("~/.cache/peregrine-bench/target-reconcile/release/peregrine")
IK_BIN = os.path.expanduser(
    "~/.cache/peregrine-bench/ik_llama.cpp-479074c/build/bin/llama-bench")
MODELS = {
    "qwen05": os.path.expanduser(
        "~/.cache/peregrine-bench/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    "llama1b": os.path.expanduser(
        "~/.cache/peregrine-bench/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
}
UB = [8, 16, 32, 64, 128, 256, 512]
CHUNK = [8, 16, 32, 64, 128]
ROUNDS = 5
# CPUs 0-7 are cores 0-3 (the four full cores), 8-23 are cores 4-11 (the compact ones);
# each core has two SMT threads. One-thread rows are pinned to CPU 4 (core 2, a full core).
ONE_CPU = "4"


def say(*a):
    print(*a, file=sys.stderr, flush=True)


def loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[:3]


def timed(cmd, env=None, timeout=580):
    e = dict(os.environ)
    e.update(env or {})
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=timeout)
    wall = time.monotonic() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = ((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime))
    if p.returncode != 0:
        say("FAILED", " ".join(cmd))
        say(p.stdout[-1500:])
        say(p.stderr[-1500:])
        raise SystemExit(1)
    return p.stdout, p.stderr, wall, cpu


def fork_row(model, threads, ub, reps=1):
    """One llama-bench invocation at one ubatch. `-rtr 1` on, `-fa` at its default."""
    cmd = [IK_BIN, "-m", model, "-p", "512", "-n", "0", "-t", str(threads),
           "-rtr", "1", "-r", str(reps), "-ub", str(ub), "-o", "json"]
    if threads == 1:
        cmd = ["taskset", "-c", ONE_CPU] + cmd
    out, err, wall, cpu = timed(cmd)
    row = json.loads(out)[0]
    ts = [t for t in row.get("samples_ts", []) if t > 1.0]
    return {"engine": "ik", "model": os.path.basename(model), "threads": threads,
            "ub": ub, "samples_ts": ts, "median_ts": statistics.median(ts) if ts else None,
            "avg_ts": row.get("avg_ts"), "wall_s": round(wall, 3),
            "cpu_s": round(cpu, 3), "load": loadavg(),
            "stderr_tail": err.strip().splitlines()[-1] if err.strip() else ""}


def peregrine_row(model, threads, chunk):
    env = {"PEREGRINE_PREFILL_CHUNK": str(chunk)}
    cmd = [PEREGRINE, "bench", "-m", model, "--prompt-tokens", "512", "--tokens", "1",
           "--threads", str(threads), "--no-profile", "--json"]
    if threads == 1:
        cmd = ["taskset", "-c", ONE_CPU] + cmd
    out, err, wall, cpu = timed(cmd, env=env)
    j = json.loads(out)
    return {"engine": "peregrine", "model": os.path.basename(model), "threads": threads,
            "chunk": chunk, "prefill_tok_s": j["prefill_tok_s"],
            "prefill_s": j["prefill_s"], "decode_tok_s": j.get("decode_tok_s"),
            "wall_s": round(wall, 3), "cpu_s": round(cpu, 3), "load": loadavg()}


def run_series(name, res):
    path = os.path.join(res, f"{name}-{os.environ.get('ONLY', 'both')}{os.environ.get('TAG', '')}.jsonl")
    with open(path, "w") as fh:
        def emit(row):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            say("   " + json.dumps({k: v for k, v in row.items()
                                    if k in ("engine", "model", "threads", "ub", "chunk",
                                             "median_ts", "prefill_tok_s", "wall_s", "cpu_s")}))

        if name == "clock":
            # The one calibration this host needs: a one-thread busy loop's wall clock
            # against its own CPU time, and against the timestamp protocol.
            src = os.path.join(res, "clockprobe.c")
            exe = os.path.join(res, "clockprobe")
            subprocess.run(["gcc", "-O2", src, "-o", exe], check=True)
            for _ in range(3):
                out, _, wall, cpu = timed(["taskset", "-c", ONE_CPU, exe, "3"])
                emit({"engine": "clock", "wall_s": round(wall, 3), "cpu_s": round(cpu, 3),
                      "probe": out.strip(), "load": loadavg()})
            return path

        points = ([{"ub": u} for u in UB] if name.startswith("fork")
                  else [{"chunk": c} for c in CHUNK])
        threads = 1 if name.endswith("1t") else 12
        only = os.environ.get("ONLY")
        tags = ("qwen05", "llama1b") if not only else (only,)
        for rnd in range(ROUNDS):
            for p in points:
                for tag in tags:
                    model = MODELS[tag]
                    if name.startswith("fork"):
                        row = fork_row(model, threads, p["ub"])
                    else:
                        row = peregrine_row(model, threads, p["chunk"])
                    row["round"] = rnd
                    emit(row)
    say(f"wrote {path}")
    return path


def main():
    res = sys.argv[1]
    os.makedirs(res, exist_ok=True)
    for s in sys.argv[2:]:
        say(f"== {s} (load {loadavg()}) {datetime.now().isoformat(timespec='seconds')}")
        run_series(s, res)


if __name__ == "__main__":
    main()
