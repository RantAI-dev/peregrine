#!/bin/sh
# One interleaved round of the S5.4h table: the exact default, the fast mode of the
# *before* build (S5.4f's table), and the fast mode of the *after* build (Q5_0
# joined), in that order, so a drift inside the round hits all three alike.
#
#   ./pg-round.sh <model-file> <prompt-tokens> <threads> <round> <cpus>
#
# For 10 threads `<cpus>` is 5-9,15-19 (the ten Cortex-X925 cores); for one thread
# it is 5. Result: one JSON line per cell in `runs.jsonl`, the 1-minute load
# average beside every line.
set -e
model=$1
prompt=$2
threads=$3
round=$4
cpus=$5

after=target/release/peregrine
before=/var/tmp/pg-s54h-before-target/release/peregrine
here=$(dirname "$0")
json="$here/runs.jsonl"

record() { # mode
    load=$(cut -d' ' -f1 /proc/loadavg)
    python3 - "$json" "$round" "$1" "$prompt" "$threads" "$load" "$model" <<'PY'
import json, sys
out, rnd, mode, prompt, threads, load, model = sys.argv[1:]
d = json.load(open("/tmp/s5.4h.json"))
rec = {
    "round": int(rnd), "mode": mode,
    "prompt_tokens": int(prompt), "threads": int(threads), "load1": float(load),
    "prefill_tok_s": d["prefill_tok_s"], "prefill_s": d["prefill_s"],
    "model": model.rsplit("/", 1)[-1],
}
open(out, "a").write(json.dumps(rec) + "\n")
print(f'{mode:11} {prompt:>4} tok {threads:>2} th  {d["prefill_tok_s"]:8.1f} tok/s  load {load}')
PY
}

taskset -c "$cpus" "$after" bench -m "$model" --prompt-tokens "$prompt" --tokens 1 \
    --threads "$threads" --json > /tmp/s5.4h.json
record exact
taskset -c "$cpus" "$before" bench -m "$model" --prompt-tokens "$prompt" --tokens 1 \
    --threads "$threads" --fast-prompt --json > /tmp/s5.4h.json
record fast-before
taskset -c "$cpus" "$after" bench -m "$model" --prompt-tokens "$prompt" --tokens 1 \
    --threads "$threads" --fast-prompt --json > /tmp/s5.4h.json
record fast-after
