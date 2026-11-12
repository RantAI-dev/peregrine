#!/usr/bin/env python3
"""S5.2c: the before/after driver, interleaved in one process sequence.

    python3 ab.py <out.jsonl> <lens> <threads> [models] [decode] [slots] [prof]

Runs two `promptbench` builds that differ only in the change under test —
`$PB_BEFORE` and `$PB_AFTER`, by default the `pb-before` and `pb-after` binaries of
this round — alternating A, B, A, B ... within a round, so a slow patch of the host
lands on both arms. Records wall, CPU and the load average beside each cell.
`REVERSE=1` flips the arm order (the drift check). ROUNDS sets the round count.
"""

import json
import os
import resource
import subprocess
import sys
import time

SCRATCH = os.path.expanduser("~/.cache/peregrine-bench/s52c")
ARMS = {
    "before": os.environ.get("PB_BEFORE", os.path.join(SCRATCH, "pb-before")),
    "after": os.environ.get("PB_AFTER", os.path.join(SCRATCH, "pb-after")),
}
CORPUS = os.path.join(SCRATCH, "corpus.txt")
# The two files of every earlier desktop round, by their regular paths (the
# `~/.cache/peregrine-bench/models/` entries are symlinks).
MODELS = {
    "llama1b": os.path.expanduser("~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "qwen05": os.path.expanduser("~/models-ab/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
}


def load():
    return " ".join(open("/proc/loadavg").read().split()[:3])


def run(bin, model, length, threads, decode, slots, prof):
    cmd = [bin, model, CORPUS, str(length), str(threads),
           "--decode", str(decode), "--slots", str(slots)]
    if threads == 1 and slots == 1:
        cmd = ["taskset", "-c", "4"] + cmd
    e = dict(os.environ)
    e.pop("PEREGRINE_PROF2", None)
    e["PEREGRINE_PROF2"] = prof
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=580)
    wall = time.monotonic() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    if p.returncode:
        raise SystemExit(f"FAILED {cmd}\n{p.stdout[-800:]}\n{p.stderr[-800:]}")
    row = json.loads(p.stdout.strip().splitlines()[-1])
    row["_wall_s"] = round(wall, 2)
    row["_cpu_s"] = round(cpu, 2)
    return row


def main():
    outp = sys.argv[1]
    lens = [int(x) for x in sys.argv[2].split(",")]
    threads = [int(x) for x in sys.argv[3].split(",")]
    which = sys.argv[4] if len(sys.argv) > 4 else "llama1b,qwen05"
    decode = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    slots = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    prof = sys.argv[7] if len(sys.argv) > 7 else "1"
    mods = [m.strip() for m in which.split(",")]
    arms = dict(ARMS)
    if os.environ.get("REVERSE"):
        arms = dict(reversed(list(arms.items())))
    out = open(outp, "a")
    for rnd in range(int(os.environ.get("ROUNDS", "3"))):
        for tag in mods:
            for ln in lens:
                for th in threads:
                    rec = {"round": rnd, "model": tag, "len": ln, "threads": th,
                           "decode": decode, "slots": slots, "load": load(), "arms": {}}
                    for name, bin in arms.items():
                        rec["arms"][name] = run(bin, MODELS[tag], ln, th, decode, slots, prof)
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    a = rec["arms"]
                    print(f"r{rnd} {tag} L{ln} t{th} before {a['before']['prefill_tok_s']:.1f}"
                          f" after {a['after']['prefill_tok_s']:.1f}"
                          f" ratio {a['after']['prefill_tok_s'] / a['before']['prefill_tok_s']:.3f}"
                          f" load {rec['load']}", flush=True)


if __name__ == "__main__":
    main()
