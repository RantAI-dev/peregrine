#!/usr/bin/env python3
"""S5.2c: two configurations of the same build, interleaved per round.

    python3 cfgab.py <out.jsonl> <lens> <threads> <ENV_A=1,ENV_B=2>

`ENV` is `NAME=value:NAME=value` pairs, A then B. Both arms run the same
`promptbench` binary; the arms differ only in the environment variables, so any
difference is the switch. Records wall, CPU and the load average beside each cell.
"""

import json
import os
import sys
import time
import subprocess

SCRATCH = os.path.expanduser("~/.cache/peregrine-bench/s52c")
BIN = os.environ.get("PB", os.path.join(SCRATCH, "pb-b0"))
CORPUS = os.path.join(SCRATCH, "corpus.txt")
MODELS = {
    "llama1b": os.path.expanduser("~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "qwen05": os.path.expanduser("~/models-ab/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
}


def load():
    return " ".join(open("/proc/loadavg").read().split()[:3])


def run(env_extra, model, length, threads):
    cmd = [BIN, model, CORPUS, str(length), str(threads), "--decode", "0"]
    e = dict(os.environ)
    e.pop("PEREGRINE_PROF2", None)
    e["PEREGRINE_PROF2"] = "1"
    e.update(env_extra)
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=580)
    wall = time.monotonic() - t0
    if p.returncode:
        raise SystemExit(f"FAILED {cmd}\n{p.stdout[-600:]}\n{p.stderr[-600:]}")
    row = json.loads(p.stdout.strip().splitlines()[-1])
    row["_wall_s"] = round(wall, 2)
    return row


def main():
    outp, lens, threads, spec = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    which = sys.argv[5] if len(sys.argv) > 5 else "llama1b,qwen05"
    a_s, b_s = spec.split(",", 1)

    def parse(s):
        d = {}
        for kv in s.split(":"):
            if kv:
                k, v = kv.split("=", 1)
                d[k] = v
        return d

    arms = {"a": parse(a_s), "b": parse(b_s)}
    out = open(outp, "a")
    for rnd in range(int(os.environ.get("ROUNDS", "3"))):
        for tag in which.split(","):
            for ln in [int(x) for x in lens.split(",")]:
                for th in [int(x) for x in threads.split(",")]:
                    rec = {"round": rnd, "model": tag, "len": ln, "threads": th,
                           "load": load(), "arms": {}}
                    for name, env in arms.items():
                        rec["arms"][name] = run(env, MODELS[tag], ln, th)
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    x, y = rec["arms"]["a"], rec["arms"]["b"]
                    print(f"r{rnd} {tag} L{ln} t{th} a {x['prefill_tok_s']:.1f}"
                          f" b {y['prefill_tok_s']:.1f} b/a {y['prefill_tok_s'] / x['prefill_tok_s']:.3f}"
                          f" load {rec['load']}", flush=True)


if __name__ == "__main__":
    main()
