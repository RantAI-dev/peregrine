#!/bin/bash
# One account cell of the S5.4i report: one engine mode, one model, one prompt length,
# one thread count.
#
#   account-cell.sh <mode pg-exact|pg-fast> <model.gguf> <prompt tokens> <threads> <out.jsonl>
#
# `PEREGRINE_PROF2=full` on every run, so the JSON carries the caller's `StepStats`
# buckets and the `peregrine_core::prof` slots (the weight matmul by matrix group and by
# weight format, attention's three stages, norms, quantise, RoPE, KV write, residual,
# SwiGLU, barrier wait).
#
# Quietness is recorded two ways: the 1-minute load average (which on this WSL2 VM is
# stale and rises with the benchmark's own 12-thread arms, so it is a poor metric) and
# the CPU-seconds spent on this VM outside the pinning set during the run, computed from
# `/proc/stat` (`other_busy_s`). For a one-thread cell every CPU but cpu0 is outside the
# set; a 12-thread cell uses the whole 0-23 set, so it records -1.
set -u
MODE=$1
M=$2
P=$3
T=$4
OUT=$5

PB=~/peregrine/target/release/examples/promptbench
CORPUS=~/.cache/peregrine-bench/s52b/corpus.txt
CPUS=0
OUTSIDE=1-23
if [ "$T" -gt 1 ]; then CPUS=0-23; OUTSIDE=""; fi

l=$(cut -d' ' -f1 /proc/loadavg)

read -r busy0 idle0 < <(python3 - "$OUTSIDE" <<'PY'
import sys
sel = sys.argv[1]
want = set()
if sel:
    for part in sel.split(','):
        if '-' in part:
            a, b = part.split('-')
            want.update(range(int(a), int(b) + 1))
        else:
            want.add(int(part))
busy = idle = 0
for line in open('/proc/stat'):
    f = line.split()
    if not f[0].startswith('cpu') or f[0] == 'cpu':
        continue
    n = int(f[0][3:])
    v = [int(x) for x in f[1:]]
    if want and n not in want:
        continue
    busy += sum(v) - v[3] - v[4]
    idle += v[3] + v[4]
print(busy, idle)
PY
)

case "$MODE" in
    pg-exact)
        out=$(PEREGRINE_PROF2=full taskset -c "$CPUS" "$PB" "$M" "$CORPUS" "$P" "$T" 2>/dev/null)
        ;;
    pg-fast)
        out=$(PEREGRINE_FAST_PROMPT=1 PEREGRINE_PROF2=full taskset -c "$CPUS" "$PB" "$M" "$CORPUS" "$P" "$T" 2>/dev/null)
        ;;
    *)
        echo "unknown mode $MODE" >&2
        exit 2
        ;;
esac

read -r busy1 idle1 < <(python3 - "$OUTSIDE" <<'PY'
import sys
sel = sys.argv[1]
want = set()
if sel:
    for part in sel.split(','):
        if '-' in part:
            a, b = part.split('-')
            want.update(range(int(a), int(b) + 1))
        else:
            want.add(int(part))
busy = idle = 0
for line in open('/proc/stat'):
    f = line.split()
    if not f[0].startswith('cpu') or f[0] == 'cpu':
        continue
    n = int(f[0][3:])
    v = [int(x) for x in f[1:]]
    if want and n not in want:
        continue
    busy += sum(v) - v[3] - v[4]
    idle += v[3] + v[4]
print(busy, idle)
PY
)

python3 - "$MODE" "$P" "$T" "$l" "$out" "$M" "$((busy1 - busy0))" <<'PY' >>"$OUT"
import json, sys
mode, p, t, load, out, model, other = sys.argv[1:8]
d = json.loads(out)
d["mode"] = mode
d["prompt_want"] = int(p)
d["threads_want"] = int(t)
d["load1"] = float(load)
d["model"] = model.rsplit("/", 1)[-1]
d["model"] = ("llama:" if d["model"].startswith("Llama") else "qwen3:") + d["model"]
d["other_busy_s"] = int(other) / 100.0
print(json.dumps(d, sort_keys=True))
PY
