#!/bin/sh
# The five Peregrine cells of one round (S5.4e): the exact default and the fast
# mode at chunks 64, 128, 256 and 512, in that order, so a drift inside the
# round hits every mode.
#
#   ./pg-round.sh <model-file> <prompt-tokens> <threads> <round> <cpus>
set -e
model=$1
prompt=$2
threads=$3
round=$4
cpus=$5

pg=target/release/peregrine
here=$(dirname "$0")
json="$here/runs.jsonl"

record() { # mode chunk
    load=$(cut -d' ' -f1 /proc/loadavg)
    python3 - "$json" "$round" "$1" "$2" "$prompt" "$threads" "$load" "$model" <<'EOF'
import json, sys
out, rnd, mode, chunk, prompt, threads, load, model = sys.argv[1:]
d = json.load(open("/tmp/s5.4e.json"))
rec = {
    "round": int(rnd), "mode": mode,
    "chunk": None if chunk == "-" else int(chunk),
    "prompt_tokens": int(prompt), "threads": int(threads), "load1": float(load),
    "prefill_tok_s": d["prefill_tok_s"], "prefill_s": d["prefill_s"],
    "model": model.rsplit("/", 1)[-1],
}
open(out, "a").write(json.dumps(rec) + "\n")
print(f'{mode:5} chunk {chunk:>3} {prompt:>4} tok {threads:>2} th  '
      f'{d["prefill_tok_s"]:8.1f} tok/s  load {load}')
EOF
}

taskset -c "$cpus" $pg bench -m "$model" --prompt-tokens "$prompt" --tokens 1 \
    --threads "$threads" --json > /tmp/s5.4e.json
record exact -
for chunk in 64 128 256 512; do
    taskset -c "$cpus" env PEREGRINE_PREFILL_CHUNK=$chunk $pg bench -m "$model" \
        --prompt-tokens "$prompt" --tokens 1 --threads "$threads" --fast-prompt \
        --json > /tmp/s5.4e.json
    record fast "$chunk"
done
