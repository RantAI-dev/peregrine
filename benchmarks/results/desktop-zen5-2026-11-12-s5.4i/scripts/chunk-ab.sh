#!/bin/bash
# A/B of one build at two `PEREGRINE_PREFILL_CHUNK` values (S5.4i).
#
#   chunk-ab.sh <rounds> <out.jsonl> <build> <chunk-a> <chunk-b> <cell...>
set -u
N=$1
OUT=$2
BIN=$3
CA=$4
CB=$5
shift 5
for r in $(seq 1 "$N"); do
    for cell in "$@"; do
        IFS='|' read -r M P T ENVV <<<"$cell"
        for c in "$CA" "$CB"; do
            l=$(cut -d' ' -f1 /proc/loadavg)
            cpus=0
            [ "$T" -eq 1 ] || cpus=0-23
            v=$(env $ENVV PEREGRINE_PREFILL_CHUNK="$c" taskset -c "$cpus" "$BIN" bench -m "$M" \
                --prompt-tokens "$P" --tokens 1 --threads "$T" --json --fast-prompt \
                2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin)["prefill_tok_s"])')
            python3 -c "
import json
print(json.dumps({'chunk':$c,'model':'$(basename "$M")','p':$P,'t':$T,
                  'round':$r,'tok_s':$v,'load1':$l}))" >>"$OUT"
        done
    done
    echo "round $r done" >&2
done
