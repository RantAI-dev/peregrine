#!/bin/bash
# Interleaved rounds of the account cells: every round runs each (model, mode, length,
# threads) cell once, so a drift in the host hits every arm.
#
#   account-rounds.sh <rounds> <out.jsonl> <cpus-for-1-thread>
#
# Cells: Llama 3.2 1B Q4_K_M and Qwen3 1.7B Q4_K_M, 512 and 4096 prompt tokens, 1 and
# 12 threads, exact and fast.
set -u
N=$1
OUT=$2
S=$PWD/scripts/account-cell.sh
L=~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf
Q=/var/tmp/pg/s54c/models/Qwen3-1.7B-Q4_K_M.gguf
for r in $(seq 1 "$N"); do
    for mode in pg-exact pg-fast; do
        $S $mode "$L" 512 12 "$OUT"
        $S $mode "$L" 512 1 "$OUT"
        $S $mode "$L" 4096 12 "$OUT"
        $S $mode "$Q" 512 12 "$OUT"
        $S $mode "$Q" 512 1 "$OUT"
        $S $mode "$Q" 4096 12 "$OUT"
    done
    echo "round $r done" >&2
done
