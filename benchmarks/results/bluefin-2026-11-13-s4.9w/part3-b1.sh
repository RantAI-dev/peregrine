#!/usr/bin/env bash
# S4.9w Part 3: the single-text 128-token cells (b1), three interleaved rounds.
# usage: part3-b1.sh <round> <arm>
set -euo pipefail
round=$1
arm=$2
root=$(cd "$(dirname "$0")/../../.." && pwd)
out=$root/benchmarks/results/bluefin-2026-11-13-s4.9w/parts/part3
mkdir -p "$out"
common=(--round "$round" --threads 4 --texts-dir /var/tmp/embed-texts-s49u
        --output-dir "$out/r$round-$arm/b1" --lengths 128 --batches 1 --clients 1
        --budget 0.5 --max-requests 20 --max-big-requests 2 --cells L128_b1_c1
        --max-load 1000)
run() {
  env PEREGRINE_EMBED_CACHE_MB=0 \
    python3 "$root/benchmarks/embed.py" --select "$1" "${common[@]}" \
    --peregrine-binary "$root/target/release/peregrine" \
    --peregrine-worktree "$root" 2>&1 \
    | tee "$out/r$round-$arm-b1-$(echo "$1" | tr ':' '-').log"
}
case "$arm" in
  peregrine) for m in nomic jina; do for p in f16 q8_0; do run "$m:peregrine:$p"; done; done ;;
  competitors) for m in nomic jina; do for e in onnxruntime fastembed openvino; do
      run "$m:$e:fp32"; done; done ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac
