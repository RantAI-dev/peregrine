#!/usr/bin/env bash
# S4.9w Part 0 generation: Llama 3.2 1B Q4_K_M, 512 tokens prompt, decode.
# usage: part0-gen.sh <round> <arm>
set -euo pipefail
round=$1
arm=$2
root=$(cd "$(dirname "$0")/../../.." && pwd)
out=$root/benchmarks/results/bluefin-2026-11-13-s4.9w/parts/part0-gen
mkdir -p "$out"
case "$arm" in
  before) bin=/tmp/opencode/bin/peregrine-before; stall= ;;
  after)  bin=/tmp/opencode/bin/peregrine-after;  stall= ;;
  after_nostall) bin=/tmp/opencode/bin/peregrine-after; stall=0 ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac
cp "$bin" "$root/target/release/peregrine"
cat /proc/loadavg > "$out/r$round-$arm-load.txt"
env ${stall:+PEREGRINE_ADAPTIVE_STALL=$stall} \
  timeout 500 python3 "$root/benchmarks/run.py" \
    --models "$HOME/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
    --engines peregrine --contexts 512 --runs 1 --tokens 128 --threads 4 \
    --user-counts 1,10 --output-dir "$out/r$round-$arm" \
    > "$out/r$round-$arm.log" 2>&1
cat /proc/loadavg >> "$out/r$round-$arm-load.txt"
grep -E '^  (1 user|10 users)' "$out/r$round-$arm.log"
