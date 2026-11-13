#!/usr/bin/env bash
# S4.9w Part 1: the float path's row block, three arms (old cap, whole pass, 2 MiB bound),
# interleaved inside every round. Cells: every float encoder on this host.
# usage: part1.sh <round> [arms]
set -euo pipefail
round=$1
arms=${2:-"old whole bound"}
root=$(cd "$(dirname "$0")/../../.." && pwd)
out=$root/benchmarks/results/bluefin-2026-11-13-s4.9w/parts/part1
mkdir -p "$out"
stat0=$(awk '/^cpu /{print $2+$3+$4, $5}' /proc/stat)
for arm in $arms; do
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
  # nomic and jina: F16, the whole cell set including the mixed batch.
  for m in nomic jina; do
    run $m f16 /var/tmp/embed-texts-s49u \
      L16_b32_c1,L128_b1_c1,L128_b32_c1,L512_b1_c1,L512_b8_c1,L2000_b1_c1,mixed_b32_c1 \
      16,128,512,2000 1,8,32 --mixed-cell --mixed-batch 32
  done
  # the other float encoders this host has a float file for.
  run qwen3-embed-0.6b f16 /var/tmp/embed-texts-s49l \
    L128_b1_c1,L512_b1_c1,L2000_b1_c1 128,512,2000 1 --ctx-tokens 2048
  run bge-small f32 /var/tmp/embed-texts-s49l \
    L128_b32_c1,L512_b8_c1 128,512 8,32
  run minilm f16 /var/tmp/embed-texts-s49l \
    L128_b32_c1,L512_b8_c1 128,512 8,32
  run nemotron-3.5-embed-1b f16 /var/tmp/embed-texts-s49l \
    L512_b1_c1 512 1
done
stat1=$(awk '/^cpu /{print $2+$3+$4, $5}' /proc/stat)
echo "cpu_busy_ticks idle_ticks before: $stat0 after: $stat1" \
  > "$out/r$round-${arms// /-}-cpu.txt"
