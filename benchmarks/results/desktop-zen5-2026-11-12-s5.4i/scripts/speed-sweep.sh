#!/bin/bash
# One round of the S5.4i standings: every arm once per cell, six arms, in the same order
# every round so a drift in the host hits every engine.
#
#   speed-sweep.sh <rounds> <out.jsonl> <cell...>
#
# A cell is `model.gguf|prompt|threads|arms`, where `arms` is a comma list of
# exact,fast-base,fast-pair,ik,ik-rtr,llama. The 12-thread cells are pinned to 0-23 and
# the one-thread cells to cpu0, as S5.4c did (0-11 is six physical cores on this VM).
set -u
N=$1
OUT=$2
shift 2
IKB=~/.cache/peregrine-bench/ik_llama.cpp-479074c/build/bin/llama-bench
LCB=~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-bench

one() { # $1 = arm, $2 = model, $3 = prompt, $4 = threads
    local arm=$1 M=$2 P=$3 T=$4 cpus=0
    [ "$T" -eq 1 ] || cpus=0-23
    local v=0 tries=0
    while :; do
        case "$arm" in
            exact)
                v=$(taskset -c "$cpus" /tmp/pg-pair bench -m "$M" --prompt-tokens "$P" \
                    --tokens 1 --threads "$T" --json 2>/dev/null |
                    python3 -c 'import json,sys;print(json.load(sys.stdin)["prefill_tok_s"])')
                ;;
            fast-base)
                v=$(PEREGRINE_FAST_PROMPT=1 taskset -c "$cpus" /tmp/pg-base bench -m "$M" \
                    --prompt-tokens "$P" --tokens 1 --threads "$T" --json --fast-prompt \
                    2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["prefill_tok_s"])')
                ;;
            fast-pair)
                v=$(PEREGRINE_FAST_PROMPT=1 taskset -c "$cpus" /tmp/pg-pair bench -m "$M" \
                    --prompt-tokens "$P" --tokens 1 --threads "$T" --json --fast-prompt \
                    2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["prefill_tok_s"])')
                ;;
            ik)
                v=$(taskset -c "$cpus" "$IKB" -m "$M" -p "$P" -n 0 -t "$T" -r 1 -o json \
                    2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)[0]["avg_ts"])')
                ;;
            ik-rtr)
                v=$(taskset -c "$cpus" "$IKB" -m "$M" -p "$P" -n 0 -t "$T" -r 1 -o json -rtr \
                    2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)[0]["avg_ts"])')
                ;;
            llama)
                v=$(taskset -c "$cpus" "$LCB" -m "$M" -p "$P" -n 0 -t "$T" -r 1 -o json \
                    2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)[0]["avg_ts"])')
                ;;
            *) echo "unknown arm $arm" >&2; exit 2 ;;
        esac
        tries=$((tries + 1))
        case "$v" in
            '' | 0 | 0.0) [ "$tries" -ge 3 ] && break ;;
            *) break ;;
        esac
    done
    [ -n "$v" ] || v=0
    python3 -c "
import json
print(json.dumps({'arm':'$arm','model':'$(basename "$M")','p':$P,'t':$T,
                  'round':${ROUND:-0},'tok_s':$v,'load1':$(cut -d' ' -f1 /proc/loadavg)}))" >>"$OUT"
}

for r in $(seq 1 "$N"); do
    ROUND=$r
    for cell in "$@"; do
        IFS='|' read -r M P T ARMS <<<"$cell"
        for a in ${ARMS//,/ }; do
            one "$a" "$M" "$P" "$T"
        done
    done
    echo "round $r done" >&2
done
