#!/bin/sh
# The two other engines' cells of one round: one llama.cpp repetition and one
# ik_llama.cpp repetition at their defaults, pinned like Peregrine.
#
#   ./eng-round.sh <model-file> <prompt-tokens> <threads> <round> <cpus>
set -e
model=$1
prompt=$2
threads=$3
round=$4
cpus=$5
here=$(dirname "$0")

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
