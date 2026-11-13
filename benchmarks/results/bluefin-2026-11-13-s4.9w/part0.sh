#!/usr/bin/env bash
# S4.9w Part 0: the two pool changes on this host, interleaved BEFORE/AFTER.
# usage: part0.sh <round> <arm>   arm = before | after | after_nostall
set -euo pipefail
round=$1
arm=$2
root=$(cd "$(dirname "$0")/../../.." && pwd)
out=$root/benchmarks/results/bluefin-2026-11-13-s4.9w/parts/part0
mkdir -p "$out"
case "$arm" in
  before) bin=/tmp/opencode/bin/peregrine-before; stall= ;;
  after)  bin=/tmp/opencode/bin/peregrine-after;  stall= ;;
  after_nostall) bin=/tmp/opencode/bin/peregrine-after; stall=0 ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac
common=(--budget 0.5 --max-requests 20 --max-big-requests 2 --max-load 1000
        --threads 4 --peregrine-binary "$bin" --peregrine-worktree "$root")
stat0=$(awk '/^cpu /{print $2+$3+$4, $5}' /proc/stat)
run() { # model precision textsdir cells
  local model=$1 prec=$2 texts=$3 cells=$4
  env PEREGRINE_EMBED_CACHE_MB=0 ${stall:+PEREGRINE_ADAPTIVE_STALL=$stall} \
    python3 "$root/benchmarks/embed.py" \
      --select "$model:peregrine:$prec" --texts-dir "$texts" \
      --output-dir "$out/r$round-$arm" --round "$round" \
      --cells "$cells" "${common[@]}" 2>&1 \
    | tee "$out/r$round-$arm-$model.log"
}
run nomic       f16  /var/tmp/embed-texts-s49u L128_b1_c1,L128_b8_c1,L128_b32_c1,L16_b8_c1,L16_b32_c1,L512_b8_c1
run jina        f16  /var/tmp/embed-texts-s49u L128_b1_c1,L128_b8_c1,L128_b32_c1,L16_b8_c1,L16_b32_c1,L512_b8_c1
run bge-small   f32  /var/tmp/embed-texts      L128_b32_c1
run minilm      f16  /var/tmp/embed-texts      L128_b32_c1
run qwen3-embed-0.6b f16 /var/tmp/embed-texts L512_b1_c1
stat1=$(awk '/^cpu /{print $2+$3+$4, $5}' /proc/stat)
echo "cpu_busy_ticks idle_ticks before: $stat0 after: $stat1" \
  > "$out/r$round-$arm-cpu.txt"
