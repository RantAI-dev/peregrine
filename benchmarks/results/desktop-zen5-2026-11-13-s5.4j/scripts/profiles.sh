#!/bin/bash
# The S5.4j account and the two cells that must not move: every measurement runs the same
# promptbench binary against the two builds that differ only in this step's code.
#
#   profiles.sh <promptbench binary> <model>
#
# Writes raw/account-<cell>-<build>.txt (the JSON line and the pool's log) and
# raw/decode-<build>.txt, raw/tenusers-<build>.txt.
set -u
PB=$1; M=$2
CORPUS=~/.cache/peregrine-bench/s52b/corpus.txt
for b in base final; do
    BIN=/tmp/pg-s54j-base; [ "$b" = final ] && BIN=/tmp/pg-s54j-final
    for cell in 512 4096; do
        PEREGRINE_FAST_PROMPT=1 PEREGRINE_PROF2=full PEREGRINE_ADAPTIVE_LOG=1 \
            taskset -c 0-23 "$PB" "$M" "$CORPUS" "$cell" 12 > "raw/account-$cell-12-$b.txt" 2>&1
    done
    # Decode 32 tokens after each prompt, and ten users on one pool.
    PEREGRINE_FAST_PROMPT=1 taskset -c 0-23 "$PB" "$M" "$CORPUS" 512 12 --decode 32 \
        > "raw/decode-$b.txt" 2>&1
    PEREGRINE_FAST_PROMPT=1 taskset -c 0-23 "$PB" "$M" "$CORPUS" 512 12 --slots 10 \
        > "raw/tenusers-$b.txt" 2>&1
    echo "$b done" >&2
done
