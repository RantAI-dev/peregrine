#!/bin/sh
# The S5.6 account: the pool's own join counters for one 4 096-token prefill at ten
# threads, quiet and under a controlled disturbance.
#
#   ./account.sh <promptbench> <label> [levels]
#
# `promptbench` is a release build of the example with `arch-llama` and `arch-qwen`.
# Run it once for the `before` binary and once for the `after` binary; the two
# binaries are compared by the README's account table. Appends to raw/account.txt.
# Every disturber is stopped by PID in the EXIT trap; nothing is matched by pattern.
set -e
bin=$1
label=$2
levels=${3:-"0 1 4"}
here=$(dirname "$0")
root=$(cd "$here/../../.." && pwd)
corpus=${CORPUS:-/tmp/opencode/corpus-long.txt}
model=${MODEL:-$HOME/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf}
out="$here/raw/account.txt"
mkdir -p "$here/raw"
: >> "$out"
pid=
cleanup() { [ -n "$pid" ] && kill "$pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for d in $levels; do
    if [ "$d" -gt 0 ]; then
        taskset -c 5-9,15-19 "$root/target/release/examples/loadgen" "$d" & pid=$!
        sleep 1
    fi
    load=$(cut -d' ' -f1 /proc/loadavg)
    PEREGRINE_PROF2=1 PEREGRINE_ADAPTIVE_LOG=1 taskset -c 5-9,15-19 "$bin" \
        "$model" "$corpus" 4096 10 > /tmp/s5.6-account.json 2> /tmp/s5.6-account.log
    {
        echo "=== $label d=$d load1=$load"
        python3 -c "import json;d=json.load(open('/tmp/s5.6-account.json'));print('wall_s',d['wall_s'],'tok_s',d['prefill_tok_s'],'barriers',d['stats_ms']['barriers'])"
        grep -E "joins:|stall|trial chose|level [0-9] smoothed" /tmp/s5.6-account.log | tail -8
    } >> "$out"
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        pid=
    fi
done
echo "appended $label to $out"
