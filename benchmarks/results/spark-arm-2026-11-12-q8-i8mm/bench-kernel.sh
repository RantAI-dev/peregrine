#!/bin/sh
# The Q8_0 tile's `sdot` body against the S4.9q `smmla` body, one thread and ten.
#
# One thread: ten runs of the kernel example pinned to X925 core 5, medians.
# Ten threads: three rounds of ten copies, one per X925 core, summed per line and
# medians over the rounds (the example is single-threaded; the aggregate is the tile's
# throughput with the cluster busy).
#
#   sh bench-kernel.sh > microbench.txt
set -eu
BIN="$HOME/wt-s49q/target/release/examples/neonbench"
CORES="5 6 7 8 9 15 16 17 18 19"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "# one thread (core 5): 10 runs"
for i in $(seq 1 10); do
  taskset -c 5 "$BIN" 4096 512 300 | grep "Q8_0 tile" >> "$OUT/st1.txt"
done
cat "$OUT/st1.txt"

echo "# ten threads: 3 rounds of 10 pinned copies, reps=100"
for r in 1 2 3; do
  for c in $CORES; do
    taskset -c "$c" "$BIN" 4096 512 100 | grep "Q8_0 tile" > "$OUT/t$r-$c.txt" &
  done
  wait
  for c in $CORES; do cat "$OUT/t$r-$c.txt"; done > "$OUT/t$r.txt"
done
cat "$OUT"/t1.txt "$OUT"/t2.txt "$OUT"/t3.txt
