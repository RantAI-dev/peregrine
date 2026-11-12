#!/bin/sh
# The S5.4k account: the fast prompt pass's own stage timers, 512 and 4 096 tokens,
# one and ten threads, both models. One JSON line per run in `account.jsonl`, the
# 1-minute load average beside every line.
#
#   ./account.sh <peregrine-bench-example> <model> <len> <threads> <runs> <tag>
#
# `<threads>` 10 pins to 5-9,15-19 (the ten Cortex-X925 cores); 1 pins to core 5.
# The example must be built with the ordinary feature set:
#   cargo build --release -p peregrine-bench --example promptbench \
#       --features arch-llama,arch-qwen,peregrine-core/fmt-legacy
set -e
bin=$1
model=$2
len=$3
threads=$4
runs=${5:-3}
tag=${6:-after}
corpus=${CORPUS:-/tmp/opencode/corpus-long.txt}
here=$(dirname "$0")

if [ "$threads" = 1 ]; then cpus=5; else cpus=5-9,15-19; fi

i=0
while [ $i -lt "$runs" ]; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    PEREGRINE_PROF2=full PEREGRINE_FAST_PROMPT=1 taskset -c "$cpus" "$bin" \
        "$model" "$corpus" "$len" "$threads" > /tmp/s5.4k-account.json
    python3 - "$here/account.jsonl" "$tag" "$model" "$len" "$threads" "$load" <<'PY'
import json, sys
out, tag, model, length, threads, load = sys.argv[1:]
d = json.load(open("/tmp/s5.4k-account.json"))
d.update(tag=tag, model=model.rsplit("/", 1)[-1], len=int(length),
         threads=int(threads), load1=float(load))
open(out, "a").write(json.dumps(d) + "\n")
print(f'{tag:8} {model.rsplit("/",1)[-1][:16]:16} {length:>5} tok {threads:>2} th '
      f'{d["prefill_tok_s"]:8.1f} tok/s  load {load}')
PY
    i=$((i + 1))
done
