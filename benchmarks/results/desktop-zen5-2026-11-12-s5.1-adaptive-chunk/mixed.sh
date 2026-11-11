#!/bin/bash
# One multi-user harness invocation: $1 model, $2 variant (before|after), $3 output dir.
set -e
export PATH="$HOME/.cargo/bin:$PATH"
if [ "$2" = before ]; then export PEREGRINE_PREFILL_CHUNK=16; else unset PEREGRINE_PREFILL_CHUNK; fi
exec timeout 560 python3 benchmarks/run.py \
  --models "$1" --engines peregrine --contexts 512 --runs 3 --tokens 64 \
  --threads 12 --user-counts 4,10 --mode server --output-dir "$3"
