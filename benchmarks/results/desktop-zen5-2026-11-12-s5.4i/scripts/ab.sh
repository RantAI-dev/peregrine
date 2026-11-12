#!/bin/bash
# An in-round alternating A/B of two builds of the same engine, for the S5.4i
# changes: every round runs each arm once, in the same order, at 1 and at 12 threads.
#
#   ab.sh <rounds> <out.jsonl> <cpu-set-12> <cell...>
#
# A cell is `model|prompt-tokens|threads|extra-env`, e.g.
#   "~/models-ab/Llama-3.2-1B-Instruct-Q4_K_M.gguf|512|12|PEREGRINE_FAST_PROMPT=1"
# Both arms run the same cell in the same round, so a drift in the host hits both.
set -u
N=$1
OUT=$2
CPUS=$3
shift 3
for r in $(seq 1 "$N"); do
    for cell in "$@"; do
        IFS='|' read -r M P T ENVV <<<"$cell"
        for arm in base pair; do
            l=$(cut -d' ' -f1 /proc/loadavg)
            cpus=$CPUS
            [ "$T" -eq 1 ] && cpus=0
            v=$(env $ENVV taskset -c "$cpus" "/tmp/pg-$arm" bench -m "$M" \
                --prompt-tokens "$P" --tokens 1 --threads "$T" --json --fast-prompt \
                2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["prefill_tok_s"])')
            python3 -c "
import json,sys
print(json.dumps({'arm':'$arm','model':'$(basename "$M")','p':$P,'t':$T,
                  'round':$r,'tok_s':$v,'load1':$l}))" >>"$OUT"
        done
    done
    echo "round $r done" >&2
done
