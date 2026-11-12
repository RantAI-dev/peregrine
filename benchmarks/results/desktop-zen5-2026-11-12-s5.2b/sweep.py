#!/usr/bin/env python3
"""S5.2b: the prompt-pass grid for both engines, one cell per process, rounds interleaved.

    python3 sweep.py <out.jsonl> <series> [models] [lens] [threads]

Series: `pgrid` (Peregrine: throughput with `PEREGRINE_PROF2=1`, plus one `=full` run per
cell for the attention split), `ikgrid` (ik_llama.cpp at its default `-ub` and at `-ub 64`),
`scaling` (Peregrine's switches at 12 threads), `decode` (single user, 64 generated
tokens), `users` (10 concurrent users). `models` is `llama1b` and/or `qwen05`; `lens` and
`threads` narrow the grid so one invocation stays under ten minutes.

Every cell records the invocation's wall and CPU time and the host's load: this host's
MONOTONIC clock has run ~10 % fast in every session since 2026-11-11
(`docs/design/s5-prefill-avx512.md` §10.0), so a row without them is not comparable.
"""

import json
import os
import resource
import statistics
import subprocess
import sys
import time

BENCH = os.path.expanduser("~/.cache/peregrine-bench/target-s52b/release/examples/promptbench")
IK = os.path.expanduser("~/.cache/peregrine-bench/ik_llama.cpp-479074c/build/bin/llama-bench")
CORPUS = os.path.expanduser("~/.cache/peregrine-bench/s52b/corpus.txt")
MODELS = {
    "llama1b": os.path.expanduser("~/.cache/peregrine-bench/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "qwen05": os.path.expanduser("~/.cache/peregrine-bench/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
}
LENS = (512, 4096)
THREADS = (1, 6, 12, 24)
ROUNDS = int(os.environ.get("ROUNDS", "3"))


def load():
    return " ".join(open("/proc/loadavg").read().split()[:3])


def timed(cmd, env=None):
    e = dict(os.environ)
    for k in ("PEREGRINE_PROF2",):
        e.pop(k, None)
    e.update(env or {})
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=580)
    wall = time.monotonic() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    if p.returncode:
        raise SystemExit(f"FAILED {cmd}\n{p.stdout[-1200:]}\n{p.stderr[-1200:]}")
    return p.stdout, p.stderr, wall, cpu


def peregrine(model, length, threads, env=None, decode=0, slots=1, chunk=None, prof=None):
    cmd = [BENCH, model, CORPUS, str(length), str(threads),
           "--decode", str(decode), "--slots", str(slots)]
    if chunk:
        cmd += ["--chunk", str(chunk)]
    if threads == 1 and slots == 1:
        cmd = ["taskset", "-c", "4"] + cmd
    e = {"PEREGRINE_PROF2": prof or "1"}
    e.update(env or {})
    out, err, wall, cpu = timed(cmd, e)
    row = json.loads(out.strip().splitlines()[-1])
    row["_wall_s"] = round(wall, 2)
    row["_cpu_s"] = round(cpu, 2)
    row["_env"] = env or {}
    row["_prof"] = e["PEREGRINE_PROF2"]
    return row


def ik(model, length, threads, ub=None, reps=5):
    cmd = [IK, "-m", model, "-p", str(length), "-n", "0", "-t", str(threads),
           "-tb", str(threads), "-rtr", "1", "-r", str(reps), "-o", "json"]
    if ub:
        cmd += ["-ub", str(ub)]
    if threads == 1:
        cmd = ["taskset", "-c", "4"] + cmd
    out, err, wall, cpu = timed(cmd)
    row = json.loads(out)[0]
    ts = [t for t in row.get("samples_ts", []) if t > 1.0]
    return {"ts": round(statistics.median(ts), 2) if ts else None,
            "samples": [round(t, 1) for t in ts],
            "ub": row.get("n_ubatch"), "_wall_s": round(wall, 2), "_cpu_s": round(cpu, 2)}


def cells(series, which, lens, threads, ub_only=None):
    models = [m.strip() for m in which.split(",")] if which else list(MODELS)
    out = []
    if series in ("pgrid", "pprof"):
        for tag in models:
            for ln in lens:
                for th in threads:
                    out.append((tag, {"len": ln, "threads": th}))
    elif series == "ikgrid":
        for tag in models:
            for ln in lens:
                for th in threads:
                    for ub in (None, 64):
                        if ub_only is not None and ub != ub_only:
                            continue
                        out.append((tag, {"len": ln, "threads": th, "ub": ub}))
    elif series == "scaling":
        th = threads[0] if threads else 12
        for tag in models:
            for ln in lens:
                for name, env, chunk in (
                    ("default", {}, None),
                    ("chunk16", {}, 16),
                    ("chunk128", {}, 128),
                    ("spin0", {"PEREGRINE_SPIN_US": "0"}, None),
                    ("spin200", {"PEREGRINE_SPIN_US": "200"}, None),
                    ("spin4000", {"PEREGRINE_SPIN_US": "4000"}, None),
                    ("affinity", {"PEREGRINE_AFFINITY": "1"}, None),
                    ("static", {"PEREGRINE_DYNAMIC": "0"}, None),
                ):
                    out.append((tag, {"len": ln, "threads": th, "tag": name,
                                      "env": env, "chunk": chunk}))
    elif series == "decode":
        for tag in models:
            for ln in lens:
                for th in threads:
                    out.append((tag, {"len": ln, "threads": th, "decode": 64,
                                      "scoring": "decode"}))
    elif series == "users":
        for tag in models:
            for ln in lens:
                for th in threads:
                    out.append((tag, {"len": ln, "threads": th, "decode": 32, "slots": 10,
                                      "scoring": "users"}))
    else:
        raise SystemExit(f"unknown series {series}")
    return out


def main():
    outp, series = sys.argv[1], sys.argv[2]
    which = sys.argv[3] if len(sys.argv) > 3 else None
    lens = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else list(LENS)
    threads = ([int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5
               else list(THREADS))
    reps = int(sys.argv[6]) if len(sys.argv) > 6 else 5
    ub_only = (None if len(sys.argv) <= 7 or sys.argv[7] == "default"
               else int(sys.argv[7]))
    out = open(outp, "a")
    plan = cells(series, which, lens, threads, ub_only)
    for rnd in range(ROUNDS):
        for tag, c in plan:
            model = MODELS[tag]
            if series == "ikgrid":
                row = ik(model, c["len"], c["threads"], c.get("ub"), reps)
                rec = {"round": rnd, "model": tag, "series": series, "load": load(), **c,
                       "ik": row}
                say = f"ik {tag} L{c['len']} t{c['threads']} ub{row['ub']} {row['ts']} tok/s"
            else:
                prof_full = "full" if series == "pprof" else None
                row = peregrine(model, c["len"], c["threads"], c.get("env"),
                                c.get("decode", 0), c.get("slots", 1), c.get("chunk"),
                                prof=prof_full)
                rec = {"round": rnd, "model": tag, "series": series, "load": load(), **c,
                       "peregrine": row}
                say = f"P {tag} L{c['len']} t{c['threads']} {c.get('tag','')} " \
                      f"{row['prefill_tok_s']:.1f} tok/s"
                if c.get("decode"):
                    say += f" dec {row['decode_tok_s']:.1f}"
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"{say}  load {rec['load']}", flush=True)


if __name__ == "__main__":
    main()
