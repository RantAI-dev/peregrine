#!/bin/bash
# The 4096-token one-thread account cells (one round): 2 models x 2 modes.
set -u
OUT=$1
N=${2:-1}
S=$PWD/scripts/account-cell.sh
L=~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf
Q=/var/tmp/pg/s54c/models/Qwen3-1.7B-Q4_K_M.gguf
for r in $(seq 1 "$N"); do
    for mode in pg-exact pg-fast; do
        $S $mode "$L" 4096 1 "$OUT"
        $S $mode "$Q" 4096 1 "$OUT"
    done
    echo "round $r done" >&2
done
