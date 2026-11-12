#!/bin/sh
# One interleaved round of the S5.4k table: Peregrine's exact and fast arms, then
# the two other engines at their defaults. One JSON line per Peregrine cell in
# `runs.jsonl` (the 1-minute load average beside it); llama.txt/ik.txt carry the
# engines' blocks.
#
#   ./round.sh <model-file> <prompt-tokens> <threads> <round>
#
# 10 threads pins to 5-9,15-19 (the ten Cortex-X925 cores); 1 pins to core 5.
set -e
model=$1
prompt=$2
threads=$3
round=$4
here=$(dirname "$0")
root=$(cd "$here/../../.." && pwd)
what=${5:-both}

if [ "$threads" = 1 ]; then cpus=5; else cpus=5-9,15-19; fi

record() { # mode json-file
    load=$(cut -d' ' -f1 /proc/loadavg)
    python3 - "$here/runs.jsonl" "$round" "$1" "$prompt" "$threads" "$load" "$model" "$2" <<'PY'
import json, sys
out, rnd, mode, prompt, threads, load, model, f = sys.argv[1:]
d = json.load(open(f))
rec = {"round": int(rnd), "mode": mode, "prompt_tokens": int(prompt),
       "threads": int(threads), "load1": float(load),
       "prefill_tok_s": d["prefill_tok_s"], "prefill_s": d["prefill_s"],
       "model": model.rsplit("/", 1)[-1]}
open(out, "a").write(json.dumps(rec) + "\n")
print(f'{mode:6} {prompt:>4} tok {threads:>2} th  {d["prefill_tok_s"]:8.1f} tok/s  load {load}')
PY
}

taskset -c "$cpus" "$root/target/release/peregrine" bench -m "$model" --prompt-tokens "$prompt" \
    --tokens 1 --threads "$threads" --json > /tmp/s5.4k-exact.json
record exact /tmp/s5.4k-exact.json
taskset -c "$cpus" "$root/target/release/peregrine" bench -m "$model" --prompt-tokens "$prompt" \
    --tokens 1 --threads "$threads" --fast-prompt --json > /tmp/s5.4k-fast.json
record fast /tmp/s5.4k-fast.json

if [ "$what" = pg ]; then echo "peregrine recorded"; exit 0; fi
load=$(cut -d' ' -f1 /proc/loadavg)
{
    echo "=== round $round, prompt $prompt, threads $threads, load1 $load"
    CUDA_VISIBLE_DEVICES="" taskset -c "$cpus" \
        ~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-bench \
        -m "$model" -p "$prompt" -n 0 -t "$threads" -r 1 2>&1 | tail -3
} >> "$here/llama.txt"
[ "$what" = pg ] && { echo "peregrine recorded"; exit 0; }
load=$(cut -d' ' -f1 /proc/loadavg)
{
    echo "=== round $round, prompt $prompt, threads $threads, load1 $load"
    taskset -c "$cpus" ~/.cache/peregrine-bench/ik_llama.cpp/build/bin/llama-bench \
        -m "$model" -p "$prompt" -n 0 -t "$threads" -r 1 2>&1 | tail -3
} >> "$here/ik.txt"
echo "engines recorded"
