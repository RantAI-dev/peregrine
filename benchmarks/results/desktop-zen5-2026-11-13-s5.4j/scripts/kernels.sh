#!/bin/bash
# The kernel bench (S5.4j): both attention orders on the same cache, one core and twelve.
set -u
ATT=${ATT:-target/release/examples/attnbench}
M=$1
for ctx in 512 4096; do
    for t in 1 12; do
        for f in 0 1; do
            if [ "$f" = 1 ]; then export PEREGRINE_FAST_ATTN=1; else unset PEREGRINE_FAST_ATTN; fi
            echo -n "ctx $ctx t $t fast $f: "
            taskset -c 0-23 "$ATT" "$M" "$ctx" "$t" 2 128 | head -1
        done
    done
done
