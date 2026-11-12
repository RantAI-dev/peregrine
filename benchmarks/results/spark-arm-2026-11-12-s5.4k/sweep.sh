#!/bin/sh
# The two sweeps of the S5.4k account, recorded with the 1-minute load average:
#   chunk 64/128/256/512 at 4 096 tokens, and thread scaling 1/2/4/6/8/10.
# One line per cell in `sweeps.txt`; the JSON is in /tmp/s5.4k-sweep.json.
#
#   ./sweep.sh <model> <cell> <threads|chunk-list> <kind:chunk|threads>
set -e
model=$1
len=$2
arg=$3
kind=$4
here=$(dirname "$0")
root=$(cd "$here/../../.." && pwd)
corpus=${CORPUS:-/tmp/opencode/corpus-long.txt}

cpus_for() {
    case $1 in
        1) echo 5 ;; 2) echo 5,6 ;; 4) echo 5-8 ;; 6) echo 5-9,15 ;; 8) echo 5-9,15-17 ;;
        *) echo 5-9,15-19 ;;
    esac
}

load=$(cut -d' ' -f1 /proc/loadavg)
if [ "$kind" = chunk ]; then
    for ch in 64 128 256 512; do
        PEREGRINE_FAST_PROMPT=1 PEREGRINE_PREFILL_CHUNK=$ch taskset -c 5 \
            "$root/target/release/examples/promptbench" "$model" "$corpus" "$len" 1 \
            > /tmp/s5.4k-sweep.json
        python3 -c "import json;d=json.load(open('/tmp/s5.4k-sweep.json'));print('chunk $ch ${len}/1 %8.1f tok/s  matmul %6.2f s  attn %6.2f s' % (d['prefill_tok_s'], d['stats_ms']['matmul']/1e3, d['stats_ms']['attention']/1e3))" | tee -a "$here/sweeps.txt"
    done
else
    for t in $arg; do
        PERGEGRINE_PREFILL_CHUNK= PEREGRINE_FAST_PROMPT=1 taskset -c "$(cpus_for $t)" \
            "$root/target/release/examples/promptbench" "$model" "$corpus" "$len" "$t" \
            > /tmp/s5.4k-sweep.json
        python3 -c "import json;d=json.load(open('/tmp/s5.4k-sweep.json'));print('threads $t ${len} %8.1f tok/s  matmul %6.2f s  attn %6.2f s' % (d['prefill_tok_s'], d['stats_ms']['matmul']/1e3, d['stats_ms']['attention']/1e3))" | tee -a "$here/sweeps.txt"
    done
fi
echo "load1 $load" | tee -a "$here/sweeps.txt"
