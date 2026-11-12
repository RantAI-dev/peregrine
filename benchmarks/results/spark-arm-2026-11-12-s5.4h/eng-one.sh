#!/bin/sh
# One engine's cell of one round, when the pair does not fit a foreground command.
#   ./eng-one.sh <llama|ik> <model-file> <prompt-tokens> <threads> <round> <cpus>
set -e
engine=$1; model=$2; prompt=$3; threads=$4; round=$5; cpus=$6
here=$(dirname "$0")
load=$(cut -d' ' -f1 /proc/loadavg)
if [ "$engine" = llama ]; then
    out="$here/llama.txt"
    bin=~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-bench
else
    out="$here/ik.txt"
    bin=~/.cache/peregrine-bench/ik_llama.cpp/build/bin/llama-bench
fi
{
    echo "=== round $round, prompt $prompt, threads $threads, load1 $load"
    CUDA_VISIBLE_DEVICES="" taskset -c "$cpus" "$bin" \
        -m "$model" -p "$prompt" -n 0 -t "$threads" -r 1 2>&1 | tail -3
} >> "$out"
echo "$engine done"
