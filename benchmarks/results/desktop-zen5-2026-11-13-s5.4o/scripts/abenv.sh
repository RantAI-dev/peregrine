#!/bin/bash
# Alternating A/B of two environment settings on one binary (S5.4o). Every round runs
# both arms in the same order on the same cells, so a drift in the host hits both.
#
#   abenv.sh <rounds> <out.jsonl> <cpu-set> <env-a> <env-b> <cell...>
#
# A cell is `model.gguf|prompt|threads`. `env-a`/`env-b` are space-separated assignments
# (`-` for none). MODE=exact times the exact engine instead of the fast prompt mode.
# PG overrides the binary (default /tmp/pg-s54o-base).
set -u
N=$1; OUT=$2; CPUS=$3; EA=$4; EB=$5; shift 5
MODE=${MODE:-fast}
PG=${PG:-/tmp/pg-s54o-base}
one() { # $1 = arm, $2 = env, $3 = model, $4 = prompt, $5 = threads
    local arm=$1 envs=$2 M=$3 P=$4 T=$5 v
    [ "$envs" = "-" ] && envs=""
    if [ "$MODE" = exact ]; then
        v=$(env $envs taskset -c "$CPUS" "$PG" bench -m "$M" --prompt-tokens "$P" \
            --tokens 1 --threads "$T" --json 2>/dev/null |
            python3 -c 'import json,sys;print(json.load(sys.stdin)["prefill_tok_s"])')
    else
        v=$(env $envs PEREGRINE_FAST_PROMPT=1 taskset -c "$CPUS" "$PG" bench -m "$M" \
            --prompt-tokens "$P" --tokens 1 --threads "$T" --json --fast-prompt \
            2>/dev/null |
            python3 -c 'import json,sys;print(json.load(sys.stdin)["prefill_tok_s"])')
    fi
    python3 -c "
import json
print(json.dumps({'arm':'$arm','model':'$(basename "$M")','p':$P,'t':$T,
                  'round':${ROUND:-0},'tok_s':$v,'load1':$(cut -d' ' -f1 /proc/loadavg)}))" >>"$OUT"
}
for r in $(seq 1 "$N"); do
    ROUND=$r
    for cell in "$@"; do
        IFS='|' read -r M P T <<<"$cell"
        one a "$EA" "$M" "$P" "$T"
        one b "$EB" "$M" "$P" "$T"
    done
    echo "round $r done" >&2
done
