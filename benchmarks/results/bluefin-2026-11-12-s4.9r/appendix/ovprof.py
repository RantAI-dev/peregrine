import sys
from collections import defaultdict
import numpy as np
import openvino as ov

path, n = sys.argv[1], int(sys.argv[2])
core = ov.Core()
model = core.read_model(path)
model.reshape({i.get_any_name(): ov.PartialShape([1, n]) for i in model.inputs})
compiled = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 4, "PERF_COUNT": True})
ids = np.zeros((1, n), dtype=np.int64); mask = np.ones((1, n), dtype=np.int64)
request = compiled.create_infer_request()
inputs = {i.get_any_name(): v for i, v in zip(model.inputs, [ids, mask, ids])}
request.infer(inputs)
request.infer(inputs)   # warm
info = request.get_profiling_info()
by_type = defaultdict(float); count = defaultdict(int)
rows = []
for node in info:
    ms = node.real_time.total_seconds() * 1e3
    by_type[node.node_type] += ms; count[node.node_type] += 1
    rows.append((ms, node.node_type, node.node_name))
total = sum(by_type.values())
print(f"# {path.split('/')[-2]} n={n} warm, {len(info)} nodes, total {total:.2f} ms")
for t, ms in sorted(by_type.items(), key=lambda kv: -kv[1]):
    if ms > 0.001:
        print(f"  {t:<22} {ms:8.3f} ms {100*ms/total:5.1f}%  ({count[t]} nodes)")
print("  -- nodes > 0.05 ms --")
for ms, t, nm in sorted(rows, reverse=True):
    if ms > 0.05:
        print(f"    {ms:8.3f}  {t:<18} {nm}")
