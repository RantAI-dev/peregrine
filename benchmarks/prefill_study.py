#!/usr/bin/env python3
"""The prefill study's measurement driver (S5, desktop Zen 5, 2026-11-11).

Every series writes one JSON file under the results directory given on the command
line. Peregrine rows come from `peregrine bench --json` (plus the `PEREGRINE_PROF2`
phase dump on stderr when a series asks for it); ik_llama.cpp rows come from
`llama-bench -o json`.

    python3 benchmarks/prefill_study.py <results-dir> <series> [...]

Every series runs its points interleaved (one round = every point once) so a slow
patch of the host lands on all of them. Series: peregrine-threads,
peregrine-chunk, peregrine-chunk-1t, peregrine-krepack, peregrine-isa,
peregrine-pin, peregrine-profile, ik-threads, ik-ub, ik-rtr.
Nothing here touches the engine; it only runs the two binaries.
"""

import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEREGRINE = os.path.join(ROOT, "target/release/peregrine")
IK_BIN = os.path.expanduser(
    "~/.cache/peregrine-bench/ik_llama.cpp-479074c/build/bin/llama-bench"
)

MODELS = {
    "qwen05": os.path.expanduser(
        "~/.cache/peregrine-bench/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    "llama1b": os.path.expanduser(
        "~/.cache/peregrine-bench/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
}

RUNS = 3


def uptime():
    with open("/proc/loadavg") as f:
        return f.read().split()[:3]


def say(*a):
    print(*a, file=sys.stderr, flush=True)


def run(cmd, env=None, timeout=580):
    e = dict(os.environ)
    if env:
        e.update(env)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=timeout)
    dt = time.time() - t0
    if p.returncode != 0:
        say("FAILED", " ".join(cmd))
        say(p.stdout[-2000:])
        say(p.stderr[-2000:])
        raise SystemExit(1)
    return p.stdout, p.stderr, dt


def peregrine(model, threads, prompt=512, tokens=1, env=None, cpus=None):
    cmd = [PEREGRINE, "bench", "-m", model, "--prompt-tokens", str(prompt),
           "--tokens", str(tokens), "--threads", str(threads), "--no-profile", "--json"]
    if cpus:
        cmd = ["taskset", "-c", cpus] + cmd
    out, err, dt = run(cmd, env=env)
    j = json.loads(out)
    j["_wall_s"] = round(dt, 3)
    j["_env"] = {k: v for k, v in (env or {}).items() if k != "PEREGRINE_PROF2"}
    j["_taskset"] = cpus
    j["_prof"] = err if (env or {}).get("PEREGRINE_PROF2") else ""
    return j


def ik(model, extra, reps=5):
    """One llama-bench invocation. Its `avg_ts` is a *mean* and WSL2's clock jumps make
    single samples meaningless (a run can read a negative time), so every row also gets
    `_median_ts`, the median of its per-repetition samples."""
    cmd = [IK_BIN, "-m", model, "-p", "512", "-n", "0", "-r", str(reps), "-o", "json"] + extra
    out, err, dt = run(cmd)
    rows = json.loads(out)
    for r in rows:
        r["_wall_s"] = round(dt, 3)
        ts = [t for t in r.get("samples_ts", []) if t > 1.0]
        r["_median_ts"] = round(statistics.median(ts), 1) if ts else None
        r["_n_samples"] = len(ts)
        r["_tag"] = (f"t{r['n_threads']}_ub{r['n_ubatch']}"
                     f"_rtr{int(r['repack'])}_fa{int(r['flash_attn'])}")
    return rows


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def peregrine_series(model_key, points, rounds=RUNS, prompt=512, label=""):
    """points: list of dicts with threads, env, cpus, tag. Points are interleaved."""
    model = MODELS[model_key]
    rows = []
    for rnd in range(rounds):
        for p in points:
            j = peregrine(model, p.get("threads", 12), prompt=prompt,
                          tokens=p.get("tokens", 1), env=p.get("env"), cpus=p.get("cpus"))
            rows.append({
                "round": rnd, "model": model_key, "tag": p.get("tag", ""),
                "threads": p.get("threads", 12), "cpus": p.get("cpus"),
                "env": p.get("env") or {},
                "prefill_tok_s": j["prefill_tok_s"], "prefill_s": j["prefill_s"],
                "prof": j["_prof"], "wall_s": j["_wall_s"],
                "bytes_acts": j["prefill_bytes_acts"],
            })
            say(f"  {label} {model_key} {p.get('tag','')} t={p.get('threads',12)} "
                f"cpus={p.get('cpus') or '-'} {j['prefill_tok_s']:.1f} tok/s")
    return rows


ALIAS = {"qwen05": ("qwen2.5-0.5b", "qwen05"), "llama1b": ("Llama-3.2-1B", "llama1b")}


def by_tag(rows, model):
    """Median prompt tok/s per tag. ik_llama.cpp rows carry `model_filename` instead."""
    out = {}
    for r in rows:
        name = r.get("model") or r.get("model_filename", "")
        if not any(a in name for a in ALIAS[model]) and r.get("model") != model:
            continue
        tag = r.get("tag") or r.get("_tag") or f"t{r.get('n_threads')}"
        v = out.setdefault(tag, [])
        v.append(r.get("_median_ts") or r.get("prefill_tok_s"))
    return {k: {"median": round(med(v), 1), "n": len(v),
                "spread": round((max(v) - min(v)) / 2, 1)} for k, v in out.items()}


def main():
    res = sys.argv[1]
    os.makedirs(res, exist_ok=True)
    model_dir = os.path.expanduser("~/.cache/peregrine-bench/models")
    os.makedirs(model_dir, exist_ok=True)
    for src in ("/home/uaa/models-ab/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                "/home/uaa/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf"):
        dst = os.path.join(model_dir, os.path.basename(src))
        if not os.path.exists(dst):
            os.symlink(src, dst)

    series = sys.argv[2:]
    for s in series:
        say(f"== {s} (load {uptime()})")
        fn = os.path.join(res, f"{s}.json")
        rows, extra = [], {}
        if s == "peregrine-threads":
            pts = [{"threads": t, "tag": f"t{t}", "env": {"PEREGRINE_PREFILL_CHUNK": "16"}}
                   for t in (1, 2, 4, 8, 12, 24)]
            for m in ("qwen05", "llama1b"):
                rows += peregrine_series(m, pts, label=s)
        elif s == "peregrine-chunk":
            pts = [{"threads": 12, "tag": f"chunk{c}", "env": {"PEREGRINE_PREFILL_CHUNK": str(c)}}
                   for c in (16, 32, 64, 128)]
            for m in ("qwen05", "llama1b"):
                rows += peregrine_series(m, pts, label=s)
        elif s == "peregrine-chunk-1t":
            pts = [{"threads": 1, "tag": f"chunk{c}", "env": {"PEREGRINE_PREFILL_CHUNK": str(c)}}
                   for c in (16, 32, 64, 128)]
            for m in ("qwen05", "llama1b"):
                rows += peregrine_series(m, pts, label=s)
        elif s == "peregrine-krepack":
            pts = [{"threads": 12, "tag": f"krepack{kp}",
                    "env": ({"PEREGRINE_KREPACK": kp} if kp != "default" else {})}
                   for kp in ("default", "0", "1")]
            for m in ("qwen05", "llama1b"):
                rows += peregrine_series(m, pts, rounds=5, label=s)
        elif s == "peregrine-isa":
            pts = [{"threads": 12, "tag": f"isa{isa}", "env": {"PEREGRINE_ISA": isa}}
                   for isa in ("avx2", "native")]
            for m in ("qwen05", "llama1b"):
                rows += peregrine_series(m, pts, rounds=5, label=s)
        elif s == "peregrine-pin":
            # CPUs 0-7 are cores 0-3 (the full cores), CPUs 8-23 are cores 4-11 (the
            # compact ones); every core has two SMT threads.
            pts = [{"threads": 4, "cpus": "0-7", "tag": "4t_on_cores0to3"},
                   {"threads": 4, "cpus": "8-15", "tag": "4t_on_cores4to7"},
                   {"threads": 4, "cpus": "16-23", "tag": "4t_on_cores8to11"},
                   {"threads": 8, "cpus": "0-7", "tag": "8t_smt_cores0to3"},
                   {"threads": 8, "cpus": "8-23", "tag": "8t_cores4to11"},
                   {"threads": 12, "cpus": None, "tag": "12t_free"},
                   {"threads": 24, "cpus": None, "tag": "24t_free",
                    "env": {"PEREGRINE_PREFILL_CHUNK": "64"}}]
            for m in ("qwen05", "llama1b"):
                rows += peregrine_series(m, pts, label=s)
        elif s == "peregrine-profile":
            # No decode token: the dump's prefill wall then equals the engine's own
            # `prefill_s` (with `--tokens 1` it does not, see the README).
            pts = [{"threads": 12, "tokens": 0, "tag": "krepack_default",
                    "env": {"PEREGRINE_PROF2": "full"}},
                   {"threads": 12, "tokens": 0, "tag": "krepack_0",
                    "env": {"PEREGRINE_PROF2": "full", "PEREGRINE_KREPACK": "0"}}]
            for m in ("qwen05", "llama1b"):
                rows += peregrine_series(m, pts, label=s)
        elif s in ("ik-threads-q", "ik-threads-l"):
            # One model per invocation: the single-thread rows are slow enough that the
            # whole sweep in one command runs past the ten-minute cap.
            m = "qwen05" if s.endswith("-q") else "llama1b"
            rows += ik(MODELS[m], ["-t", "1,2,4,8,12,24", "-tb", "1,2,4,8,12,24",
                                   "-rtr", "1", "-fa", "1"], reps=3)
        elif s == "ik-ub":
            for m in ("qwen05", "llama1b"):
                rows += ik(MODELS[m], ["-t", "12", "-tb", "12", "-rtr", "1", "-fa", "1",
                                       "-ub", "64,128,256,512"])
        elif s == "ik-rtr":
            # `-rtr` is not a swept option in llama-bench (only the first value is read),
            # so it takes one invocation per value; `-fa` does sweep.
            for m in ("qwen05", "llama1b"):
                for rtr in ("0", "1"):
                    rows += ik(MODELS[m], ["-t", "12", "-tb", "12", "-rtr", rtr, "-fa", "0,1"])
        else:
            raise SystemExit(f"unknown series {s}")
        json.dump({"rows": rows, "extra": extra,
                   "summary": {m: by_tag(rows, m) for m in ("qwen05", "llama1b")}},
                  open(fn, "w"), indent=1)
        say(f"   wrote {fn}")
    say("done")


if __name__ == "__main__":
    main()
