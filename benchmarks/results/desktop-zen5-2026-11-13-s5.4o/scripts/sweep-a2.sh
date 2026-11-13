#!/bin/bash
# One round of S5.4o's Part A grid: KV cache precisions and the two engines at the
# same cell, in the same order every round.
#
#   sweep-a2.sh <rounds> <out.jsonl> <cell...>
#
# A cell is `model.gguf|prompt|threads`. Arms:
#   exact      the default engine, f16 cache
#   fast       --fast-prompt, f16 cache
#   fast-i8    --fast-prompt, i8 cache (Hadamard-rotated)
#   exact-i8   the default engine, i8 cache
#   ik         ik_llama.cpp at its defaults (f16 cache, flash attention on)
#   ik-rtr     ik_llama.cpp with -rtr
# The Peregrine arms are `promptbench` (a real text, --decode 32) so the same run
# gives the pass and the decode after it; the fork's are its own llama-bench.
set -u
N=$1; OUT=$2; shift 2
PB=/tmp/pb-s54o-kv
IKB=~/.cache/peregrine-bench/ik_llama.cpp-479074c/build/bin/llama-bench
C=~/.cache/peregrine-bench/s52b/corpus.txt
one() { # arm, model, prompt, threads
    local arm=$1 M=$2 P=$3 T=$4 cpus=0 j
    [ "$T" -eq 1 ] || cpus=0-23
    case "$arm" in
        exact|fast|fast-i8|exact-i8)
            local envs="" kv=f16
            case "$arm" in
                fast) envs="PEREGRINE_FAST_PROMPT=1" ;;
                fast-i8) envs="PEREGRINE_FAST_PROMPT=1"; kv=i8 ;;
                exact-i8) kv=i8 ;;
            esac
            j=
            for try in 1 2 3; do
                j=$(env $envs taskset -c "$cpus" "$PB" "$M" "$C" "$P" "$T" --decode 32 --kv "$kv" 2>/dev/null)
                [ -n "$j" ] && break
            done
            ;;
        ik|ik-rtr)
            local x=""
            [ "$arm" = ik-rtr ] && x=-rtr
            j=
            for try in 1 2 3; do
                j=$(taskset -c "$cpus" "$IKB" -m "$M" -p "$P" -n 0 -t "$T" -r 1 -o json $x 2>/dev/null |
                    python3 -c 'import json,sys;d=json.load(sys.stdin)[0];print(json.dumps({"prefill_tok_s":d["avg_ts"],"decode_tok_s":0.0,"stats_ms":{"attention":0.0}}))' 2>/dev/null)
                [ -n "$j" ] && break
            done
            ;;
    esac
    python3 - "$arm" "$M" "$P" "$T" "$j" <<'EOF' >>"$OUT"
import json,sys
arm,M,P,T,j=sys.argv[1:6]
r=json.loads(j)
print(json.dumps({'arm':arm,'model':M.split('/')[-1],'p':int(P),'t':int(T),
                  'round':int(__import__('os').environ.get('ROUND','0')),
                  'tok_s':r['prefill_tok_s'],'decode_tok_s':r.get('decode_tok_s',0.0),
                  'attn_ms':r.get('stats_ms',{}).get('attention',0.0),
                  'load1':float(open('/proc/loadavg').read().split()[0])}))
EOF
}
for r in $(seq 1 "$N"); do
    export ROUND=$((r + ${OFFSET:-0}))
    for cell in "$@"; do
        IFS='|' read -r M P T <<<"$cell"
        for a in exact fast fast-i8 exact-i8 ik ik-rtr; do
            [ -n "${ARMS:-}" ] && case ",$ARMS," in *",$a,"*) ;; *) continue ;; esac
            one "$a" "$M" "$P" "$T"
        done
    done
    echo "round $r done" >&2
done
