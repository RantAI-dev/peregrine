#!/usr/bin/env python3
"""S5.2 end to end: the fork and Peregrine at their default pass widths.

One point at a time, three rounds interleaved, medians. Records wall and CPU time per
invocation because this host's clock has moved before (docs/design/s5-prefill-avx512.md
SS10.0).
"""
import json, os, resource, statistics, subprocess, sys, time

PER = os.path.expanduser("~/.cache/peregrine-bench/target-s52/release/peregrine")
IK = os.path.expanduser("~/.cache/peregrine-bench/ik_llama.cpp-479074c/build/bin/llama-bench")
MODELS = {
    "llama1b": os.path.expanduser("~/.cache/peregrine-bench/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "qwen05": os.path.expanduser("~/.cache/peregrine-bench/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
}

def timed(cmd, env=None):
    e = dict(os.environ); e.update(env or {})
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=580)
    wall = time.monotonic() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    if p.returncode: raise SystemExit(f"FAILED {cmd}\n{p.stdout[-800:]}\n{p.stderr[-800:]}")
    return p.stdout, wall, cpu

def load():
    return " ".join(open("/proc/loadavg").read().split()[:3])

def fork(model, threads, prompt, gen, reps=3):
    cmd = [IK, "-m", model, "-p", str(prompt), "-n", str(gen), "-t", str(threads),
           "-rtr", "1", "-r", str(reps), "-o", "json"]
    if threads == 1: cmd = ["taskset", "-c", "4"] + cmd
    out, wall, cpu = timed(cmd)
    row = json.loads(out)[0]
    ts = [t for t in row.get("samples_ts", []) if t > 1.0]
    return {"ts": statistics.median(ts) if ts else None, "wall_s": round(wall, 2),
            "cpu_s": round(cpu, 2)}

def peregrine(model, threads, prompt, gen, chunk=None):
    env = {"PEREGRINE_PREFILL_CHUNK": str(chunk)} if chunk else {}
    cmd = [PER, "bench", "-m", model, "--prompt-tokens", str(prompt), "--tokens", str(gen),
           "--threads", str(threads), "--no-profile", "--json"]
    if threads == 1: cmd = ["taskset", "-c", "4"] + cmd
    out, wall, cpu = timed(cmd, env=env)
    j = json.loads(out)
    return {"prompt": j.get("prefill_tok_s"), "decode": j.get("decode_tok_s"),
            "wall_s": round(wall, 2), "cpu_s": round(cpu, 2)}

def main():
    out = open(sys.argv[1], "w")
    for rnd in range(3):
        for tag, model in MODELS.items():
            for th in (1, 12):
                # Prompt processing: 512 tokens, no decode. Both at their defaults.
                r = {"round": rnd, "model": tag, "threads": th, "phase": "pp512",
                     "load": load()}
                r["fork_default"] = fork(model, th, 512, 0)
                r["peregrine_default"] = peregrine(model, th, 512, 1)
                r["peregrine_chunk64"] = peregrine(model, th, 512, 1, 64)
                out.write(json.dumps(r) + "\n"); out.flush()
                print(json.dumps(r), flush=True)
                # Decode: 128 tokens after a 16-token prompt.
                r = {"round": rnd, "model": tag, "threads": th, "phase": "tg128",
                     "load": load()}
                r["fork_default"] = fork(model, th, 16, 128)
                r["peregrine_default"] = peregrine(model, th, 16, 128)
                out.write(json.dumps(r) + "\n"); out.flush()
                print(json.dumps(r), flush=True)
main()
