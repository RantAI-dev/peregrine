#!/usr/bin/env bash
# S4.9w Part 1b: the bound's constant, measured (1 MiB vs 2 MiB vs 4 MiB) on the
# 128-token and 2 000-token cells only.
# usage: part1-side.sh <round>
set -euo pipefail
round=$1
root=$(cd "$(dirname "$0")/../../.." && pwd)
out=$root/benchmarks/results/bluefin-2026-11-13-s4.9w/parts/part1-side
mkdir -p "$out"
stat0=$(awk '/^cpu /{print $2+$3+$4, $5}' /proc/stat)
for arm in 1m bound 4m; do
  bin=/tmp/opencode/bin/pg-$arm
  run() { # model precision textsdir cells lengths batches extra...
    local model=$1 prec=$2 texts=$3 cells=$4 lengths=$5 batches=$6
    shift 6
    env PEREGRINE_EMBED_CACHE_MB=0 \
      python3 "$root/benchmarks/embed.py" \
        --select "$model:peregrine:$prec" --texts-dir "$texts" \
        --output-dir "$out/r$round-$arm" --round "$round" \
        --cells "$cells" --lengths "$lengths" --batches "$batches" \
        --clients 1 --budget 0.5 --max-requests 20 --max-big-requests 2 \
        --max-load 1000 --min-avail-gb 2 --threads 4 \
        --peregrine-binary "$bin" --peregrine-worktree "$root" "$@" 2>&1 \
      | tee "$out/r$round-$arm-$model.log"
  }
  for m in nomic jina; do
    run $m f16 /var/tmp/embed-texts-s49u \
      L128_b1_c1,L128_b32_c1,L2000_b1_c1 128,2000 1,32
  done
  run qwen3-embed-0.6b f16 /var/tmp/embed-texts-s49l \
    L128_b1_c1,L2000_b1_c1 128,2000 1 --ctx-tokens 2048
  run bge-small f32 /var/tmp/embed-texts-s49l L128_b32_c1 128 32
  run minilm f16 /var/tmp/embed-texts-s49l L128_b32_c1 128 32
done
stat1=$(awk '/^cpu /{print $2+$3+$4, $5}' /proc/stat)
echo "cpu_busy_ticks idle_ticks before: $stat0 after: $stat1" \
  > "$out/r$round-cpu.txt"
