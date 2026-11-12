#!/bin/bash
# One account cell (S5.4j): the promptbench profile of a cell, both builds, one round.
#   account.sh <out-prefix> <build> <model> <prompt> <threads> [extra promptbench args]
set -u
OUT=$1; BIN=$2; M=$3; P=$4; T=$5; shift 5
PEREGRINE_FAST_PROMPT=1 PEREGRINE_PROF2=full PEREGRINE_ADAPTIVE_LOG=1 \
    taskset -c 0-23 "$BIN" "$M" ~/.cache/peregrine-bench/s52b/corpus.txt "$P" "$T" "$@" \
    > "$OUT" 2>&1
