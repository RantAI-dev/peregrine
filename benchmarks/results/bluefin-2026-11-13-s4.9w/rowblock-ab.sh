#!/usr/bin/env bash
# S4.9w Part 2: the float path's row block, A/B (orig vs no block) through cellprof.
# usage: rowblock-ab.sh <round> <arm> <model>
set -euo pipefail
round=$1
arm=$2
model=$3
root=$(cd "$(dirname "$0")/../../.." && pwd)
out=$root/benchmarks/results/bluefin-2026-11-13-s4.9w/parts/rowblock
mkdir -p "$out"
case "$arm" in
  orig) bin=/tmp/opencode/bin/cellprof-orig ;;
  noblock) bin=/tmp/opencode/bin/cellprof-noblock ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac
env PEREGRINE_EMBED_CACHE_MB=0 taskset -c 0-3 "$bin" \
  "$HOME/models/embed/$model/$model-f16.gguf" \
  "/var/tmp/embed-texts-s49u/$model.json" 128,512 1,8,32 4 \
  > "$out/r$round-$arm-$model-128-512.jsonl" 2>/dev/null
env PEREGRINE_EMBED_CACHE_MB=0 taskset -c 0-3 "$bin" \
  "$HOME/models/embed/$model/$model-f16.gguf" \
  "/var/tmp/embed-texts-s49u/$model.json" 2000 1 4 \
  > "$out/r$round-$arm-$model-2000.jsonl" 2>/dev/null
cat "$out/r$round-$arm-$model-"*.jsonl
