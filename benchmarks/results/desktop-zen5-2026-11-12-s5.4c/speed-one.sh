#!/bin/bash
# One engine, one run, one JSONL row: the slow cells of the S5.4c speed table
# (4096 prompt tokens on one thread) do not fit five engines inside one command.
#
#   speed-one.sh <engine> <model.gguf> <prompt tokens> <threads> <cpus> <out.jsonl>
#
# engine: pg-exact | pg-fast | ik-default | ik-rtr | llama
set -u
E=$1
M=$2
P=$3
T=$4
CPUS=$5
OUT=$6

PG=~/peregrine/target/release/peregrine
IK=~/.cache/peregrine-bench/ik_llama.cpp-479074c/build/bin/llama-bench
LC=~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-bench
l=$(cut -d' ' -f1 /proc/loadavg)

case "$E" in
    pg-exact) v=$(taskset -c "$CPUS" "$PG" bench -m "$M" --prompt-tokens "$P" --tokens 1 --threads "$T" --json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["prefill_tok_s"])') ;;
    pg-fast) v=$(taskset -c "$CPUS" "$PG" bench -m "$M" --prompt-tokens "$P" --tokens 1 --threads "$T" --json --fast-prompt 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["prefill_tok_s"])') ;;
    ik-default) v=$(taskset -c "$CPUS" "$IK" -m "$M" -p "$P" -n 0 -t "$T" -r 1 -o json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["avg_ts"])') ;;
    ik-rtr) v=$(taskset -c "$CPUS" "$IK" -m "$M" -p "$P" -n 0 -t "$T" -r 1 -o json -rtr 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["avg_ts"])') ;;
    llama) v=$(taskset -c "$CPUS" "$LC" -m "$M" -p "$P" -n 0 -t "$T" -r 1 -o json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["avg_ts"])') ;;
    *) echo "unknown engine $E" >&2; exit 2 ;;
esac
printf '{"engine": "%s", "round": %s, "tok_s": %s, "load1": %s, "tries": 1}\n' "$E" "${ROUND:-0}" "$v" "$l" >>"$OUT"
echo "$E $v (load1 $l)"
