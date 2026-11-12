#!/usr/bin/env python3
"""S5.2c: Peregrine against ik_llama.cpp and llama.cpp, interleaved per round.

    python3 engines.py <out.jsonl> <lens> <threads> [rounds]

Each round runs, for every model and length, one `promptbench` cell of this branch and
one `llama-bench` cell of each engine, in the same order, so a slow patch of the host
lands on all three. `llama-bench` is given `-r 1` (one timed repetition; the round is
the repetition) and ik its fork settings of every earlier desktop round
(`-rtr 1 -ub 512`). Records the load average beside each round.
"""

import json
import os
import subprocess
import sys

SCRATCH = os.path.expanduser("~/.cache/peregrine-bench/s52c")
PB = os.path.join(SCRATCH, "pb-final")
CORPUS = os.path.join(SCRATCH, "corpus.txt")
IK = os.path.expanduser("~/.cache/peregrine-bench/ik_llama.cpp-479074c/build/bin/llama-bench")
LC = os.path.expanduser("~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-bench")
MODELS = {
    "llama1b": os.path.expanduser("~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "qwen05": os.path.expanduser("~/models-ab/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
}


def load():
    return " ".join(open("/proc/loadavg").read().split()[:3])


def bench(bin, model, n, threads, extra):
    p = subprocess.run([bin, "-m", model, "-p", str(n), "-n", "0", "-t", str(threads),
                        "-r", "1"] + extra, capture_output=True, text=True, timeout=580)
    line = [l for l in p.stdout.splitlines() if f"pp{n}" in l and "±" in l][-1]
    return float(line.split("|")[-2].split("±")[0])


def main():
    outp, lens, threads = sys.argv[1], sys.argv[2], sys.argv[3]
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    out = open(outp, "a")
    for rnd in range(rounds):
        for tag, model in MODELS.items():
            for n in [int(x) for x in lens.split(",")]:
                e = dict(os.environ)
                e["PEREGRINE_PROF2"] = "1"
                p = subprocess.run([PB, model, CORPUS, str(n), threads, "--decode", "0"],
                                   capture_output=True, text=True, env=e, timeout=580)
                d = json.loads(p.stdout.strip().splitlines()[-1])
                rec = {"round": rnd, "model": tag, "len": n, "threads": int(threads),
                       "load": load(),
                       "peregrine": d["prefill_tok_s"],
                       "ik512": bench(IK, model, n, threads, ["-rtr", "1", "-ub", "512"]),
                       "llama": bench(LC, model, n, threads, [])}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"r{rnd} {tag} L{n} P {rec['peregrine']:.1f} ik {rec['ik512']:.1f}"
                      f" llama {rec['llama']:.1f} load {rec['load']}", flush=True)


if __name__ == "__main__":
    main()
