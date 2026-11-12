#!/bin/bash
# Alternating A/B of two builds on the fast prompt mode (S5.4j). Every round runs both
# arms in the same order, on the same cells, so a drift in the host hits both.
#
#   ab.sh <rounds> <out.jsonl> <cpu-set> <build-a> <build-b> <cell...>
#
# A cell is `model.gguf|prompt|threads`. Arm `a` and `b` are the fast prompt mode; with
# `mode=exact` (env) the exact engine is timed instead, as the control that must not move.
set -u
N=$1; OUT=$2; CPUS=$3; A=$4; B=$5; shift 5
MODE=${MODE:-fast}
for r in $(seq 1 "$N"); do
    for cell in "$@"; do
        IFS='|' read -r M P T <<<"$cell"
        for arm in a b; do
            bin=$A; [ "$arm" = b ] && bin=$B
            if [ "$MODE" = exact ]; then
                v=$(taskset -c "$CPUS" "$bin" bench -m "$M" --prompt-tokens "$P" --tokens 1 \
                    --threads "$T" --json 2>/dev/null |
                    python3 -c 'import json,sys;print(json.load(sys.stdin)["prefill_tok_s"])')
            else
                v=$(PEREGRINE_FAST_PROMPT=1 taskset -c "$CPUS" "$bin" bench -m "$M" \
                    --prompt-tokens "$P" --tokens 1 --threads "$T" --json --fast-prompt \
                    2>/dev/null |
                    python3 -c 'import json,sys;print(json.load(sys.stdin)["prefill_tok_s"])')
            fi
            python3 -c "
import json
print(json.dumps({'arm':'$arm','model':'$(basename "$M")','p':$P,'t':$T,'round':$r,
                  'tok_s':$v,'load1':$(cut -d' ' -f1 /proc/loadavg)}))" >>"$OUT"
        done
    done
    echo "round $r done" >&2
done
