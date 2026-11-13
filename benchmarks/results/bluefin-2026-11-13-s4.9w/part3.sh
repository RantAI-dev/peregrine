#!/usr/bin/env bash
# S4.9w Part 3: the two families' rows, three interleaved rounds, cache off.
# usage: part3.sh <round> <arm>   arm = peregrine | competitors
set -euo pipefail
round=$1
arm=$2
root=$(cd "$(dirname "$0")/../../.." && pwd)
out=$root/benchmarks/results/bluefin-2026-11-13-s4.9w/parts/part3
mkdir -p "$out"
CELLS=L16_b8_c1,L16_b32_c1,L128_b1_c1,L128_b8_c1,L128_b32_c1,L512_b8_c1,L512_b32_c1,mixed_b32_c1
common=(--round "$round" --threads 4 --texts-dir /var/tmp/embed-texts-s49u
        --output-dir "$out/r$round-$arm/main" --lengths 16,128,512 --batches 8,32
        --clients 1 --budget 0.5 --max-requests 20 --max-big-requests 2
        --mixed-cell --mixed-batch 32 --cells "$CELLS" --max-load 1000)
run() { # select
  env PEREGRINE_EMBED_CACHE_MB=0 \
    python3 "$root/benchmarks/embed.py" --select "$1" "${common[@]}" \
    --peregrine-binary "$root/target/release/peregrine" \
    --peregrine-worktree "$root" 2>&1 \
    | tee "$out/r$round-$arm-$(echo "$1" | tr ':' '-').log"
}
case "$arm" in
  peregrine)
    for m in nomic jina; do for p in f16 q8_0; do
      run "$m:peregrine:$p"
    done; done ;;
  competitors)
    for m in nomic jina; do for e in onnxruntime fastembed openvino; do
      run "$m:$e:fp32"
    done; done ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac
