#!/bin/bash
# One speed cell of the S5.4c report: a model, a prompt length, a thread count and
# a number of interleaved rounds. Every round runs each engine once, in the same
# order, so a drift in the host hits every engine.
#
#   speed-cell.sh <model.gguf> <prompt tokens> <threads> <rounds> <out.jsonl> <cpus>
#
# Peregrine: `peregrine bench --prompt-tokens P --tokens 1 --json` (prefill_tok_s).
# ik_llama.cpp: llama-bench -p P -n 0 -t T -r 1, at its defaults and with -rtr.
# llama.cpp: the same llama-bench.
# The 1-minute load average is recorded before every run.
set -u
M=$1
P=$2
T=$3
N=$4
OUT=$5
CPUS=$6

PG=~/peregrine/target/release/peregrine
IK=~/.cache/peregrine-bench/ik_llama.cpp-479074c/build/bin/llama-bench
LC=~/.cache/peregrine-bench/llama.cpp-v0.9.0/build/bin/llama-bench

load1() { cut -d' ' -f1 /proc/loadavg; }

bench_json() { # $1 = label, rest = command
    local label=$1
    shift
    local l
    l=$(load1)
    local v=0 tries=0
    while :; do
        v=$("$@" 2>/tmp/s54c-bench.err | python3 -c '
import json,sys
d=json.load(sys.stdin)
if isinstance(d, list):
    print(d[0]["avg_ts"])
else:
    print(d["prefill_tok_s"])
' 2>/dev/null) || v=0
        tries=$((tries + 1))
        # A failed or zero run is retried (at most three times) rather than recorded:
        # llama-bench has been seen to print a zero rate once on a loaded host.
        case "$v" in
            '' | 0 | 0.0) [ "$tries" -ge 3 ] && { tail -3 /tmp/s54c-bench.err >&2; } ;;
            *) break ;;
        esac
        [ "$tries" -ge 3 ] && break
    done
    case "$v" in '' ) v=0 ;; esac
    printf '{"engine": "%s", "round": %s, "tok_s": %s, "load1": %s, "tries": %s}\n' "$label" "$ROUND" "$v" "$l" "$tries" >>"$OUT"
}

ROUND=""
for r in $(seq 1 "$N"); do
    ROUND=$r
    bench_json pg-exact taskset -c "$CPUS" "$PG" bench -m "$M" --prompt-tokens "$P" --tokens 1 --threads "$T" --json
    bench_json pg-fast taskset -c "$CPUS" "$PG" bench -m "$M" --prompt-tokens "$P" --tokens 1 --threads "$T" --json --fast-prompt
    bench_json ik-default taskset -c "$CPUS" "$IK" -m "$M" -p "$P" -n 0 -t "$T" -r 1 -o json
    bench_json ik-rtr taskset -c "$CPUS" "$IK" -m "$M" -p "$P" -n 0 -t "$T" -r 1 -o json -rtr
    bench_json llama taskset -c "$CPUS" "$LC" -m "$M" -p "$P" -n 0 -t "$T" -r 1 -o json
    echo "round $r done" >&2
done
