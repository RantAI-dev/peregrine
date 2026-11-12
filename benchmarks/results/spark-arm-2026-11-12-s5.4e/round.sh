#!/bin/sh
# One interleaved round of the S5.4e table: the exact default and the fast mode at
# chunks 64, 128, 256 and 512, then one llama.cpp and one ik_llama.cpp repetition.
#
#   ./round.sh <model-file> <prompt-tokens> <threads> <round> <cpus>
#
# For 10 threads `<cpus>` is 5-9,15-19 (the ten Cortex-X925 cores); for one
# thread it is 5. Result: one JSON line per Peregrine cell in `runs.jsonl`, and
# one block per engine in `llama.txt` / `ik.txt`. The 1-minute load average is
# recorded with every line.
set -e
model=$1
prompt=$2
threads=$3
round=$4
cpus=$5

pg=target/release/peregrine
here=$(dirname "$0")
json="$here/runs.jsonl"

record() { # mode chunk
    load=$(cut -d' ' -f1 /proc/loadavg)
    python3 - "$json" "$round" "$1" "$2" "$prompt" "$threads" "$load" "$model" <<'EOF'
import json, sys
out, rnd, mode, chunk, prompt, threads, load, model = sys.argv[1:]
d = json.load(open("/tmp/s5.4e.json"))
rec = {
    "round": int(rnd), "mode": mode,
    "chunk": None if chunk == "-" else int(chunk),
    "prompt_tokens": int(prompt), "threads": int(threads), "load1": float(load),
    "prefill_tok_s": d["prefill_tok_s"], "prefill_s": d["prefill_s"],
    "model": model.rsplit("/", 1)[-1],
}
open(out, "a").write(json.dumps(rec) + "\n")
print(f'{mode:5} chunk {chunk:>3} {prompt:>4} tok {threads:>2} th  '
      f'{d["prefill_tok_s"]:8.1f} tok/s  load {load}')
EOF
}

# The exact default, then the four fast chunks, in one round.
taskset -c "$cpus" $pg bench -m "$model" --prompt-tokens "$prompt" --tokens 1 \
    --threads "$threads" --json > /tmp/s5.4e.json
record exact -
for chunk in 64 128 256 512; do
    taskset -c "$cpus" env PEREGRINE_PREFILL_CHUNK=$chunk $pg bench -m "$model" \
        --prompt-tokens "$prompt" --tokens 1 --threads "$threads" --fast-prompt \
        --json > /tmp/s5.4e.json
    record fast "$chunk"
done

# The two other engines, one repetition each, so a drift within the round hits
# every engine alike. llama-bench warms up before its measured run.
load=$(cut -d' ' -f1 /proc/loadavg)
{
    echo "=== round $round, prompt $prompt, threads $threads, load1 $load"
    CUDA_VISIBLE_DEVICES="" taskset -c "$cpus" \
        ~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-bench \
        -m "$model" -p "$prompt" -n 0 -t "$threads" -r 1 2>&1 | tail -3
} >> "$here/llama.txt"
load=$(cut -d' ' -f1 /proc/loadavg)
{
    echo "=== round $round, prompt $prompt, threads $threads, load1 $load"
    taskset -c "$cpus" ~/.cache/peregrine-bench/ik_llama.cpp/build/bin/llama-bench \
        -m "$model" -p "$prompt" -n 0 -t "$threads" -r 1 2>&1 | tail -3
} >> "$here/ik.txt"
echo "llama.cpp and ik_llama.cpp recorded"
