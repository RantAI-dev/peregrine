"""ONNX Runtime's per-node profile of one embedding ONNX, 4 threads, one text.

The S4.9u account needs the winning engine's own profile where ONNX Runtime (or
fastembed, which is ORT) is fastest. `ORT_ENABLE_ALL`, `intra_op_num_threads=4`,
one warm-up and then one inference with `enable_profiling`; the JSON's node
durations (`dur`, microseconds) are summed by operator. Unlike OpenVINO's
PERF_COUNT real_time, ORT's `dur` is the node's wall time.

  python ortprof.py DIR/model.onnx N [out.json]
"""

import json
import sys
from collections import defaultdict

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

path, n = sys.argv[1], int(sys.argv[2])
out_path = sys.argv[3] if len(sys.argv) > 3 else ""
tok_path = path.rsplit("/", 1)[0] + "/tokenizer.json"
tok = Tokenizer.from_file(tok_path)
enc = tok.encode("x")
ids = np.full((1, n), enc.ids[0], dtype=np.int64)
mask = np.ones((1, n), dtype=np.int64)
types = np.zeros((1, n), dtype=np.int64)
so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.enable_profiling = True
sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
names = {i.name: i for i in sess.get_inputs()}
feeds = {"input_ids": ids}
if "attention_mask" in names:
    feeds["attention_mask"] = mask
if "token_type_ids" in names:
    feeds["token_type_ids"] = types
sess.run(None, feeds)
prof = sess.end_profiling()
rows = json.load(open(prof))
by = defaultdict(float)
count = defaultdict(int)
for r in rows:
    if r.get("cat") != "Node":
        continue
    op = r.get("args", {}).get("op_name") or r.get("name", "").split("_")[0]
    by[op] += r.get("dur", 0.0)
    count[op] += 1
total = sum(by.values())
print(f"# {path.split('/')[-2]} n={n} threads=4 warm, {sum(count.values())} nodes, total {total/1e3:.2f} ms")
for op, us in sorted(by.items(), key=lambda kv: -kv[1]):
    if us > 10:
        print(f"  {op:<24} {us/1e3:8.3f} ms {100*us/total:5.1f}%  ({count[op]} nodes)")
if out_path:
    json.dump({"path": path, "n": n, "total_us": total,
               "by_op_us": dict(by), "by_op_nodes": dict(count)}, open(out_path, "w"), indent=1)
