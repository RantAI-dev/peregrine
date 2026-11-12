import sys
from collections import defaultdict
import numpy as np
import openvino as ov
path, n = sys.argv[1], int(sys.argv[2])
core = ov.Core(); model = core.read_model(path)
model.reshape({i.get_any_name(): ov.PartialShape([1, n]) for i in model.inputs})
compiled = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 4, "PERF_COUNT": True})
ids = np.zeros((1, n), dtype=np.int64); mask = np.ones((1, n), dtype=np.int64)
r = compiled.create_infer_request()
ins = {i.get_any_name(): v for i, v in zip(model.inputs, [ids, mask, ids])}
r.infer(ins); r.infer(ins)
info = r.get_profiling_info()
tot = sum(x.real_time.total_seconds()*1e3 for x in info)
agg = defaultdict(float); cnt = defaultdict(int)
for x in info:
    ms = x.real_time.total_seconds()*1e3
    if x.node_type == 'Subgraph' and 'attention/self/Transpose_3' in x.node_name:
        agg['Subgraph: attention scores+softmax+mix'] += ms; cnt['a']+=1
    elif x.node_type in ('Gelu','MVN','Gather','Transpose','Convert','Broadcast','Multiply','Reshape','Where','Add','Eltwise','Parameter','Result','Concat','ShapeOf','Squeeze','Unsqueeze','MatMul','FullyConnected','Subgraph'):
        agg[f'{x.node_type}'] += ms; cnt[x.node_type]+=1
    else:
        agg['other:'+x.node_type] += ms
print(f"# {path.split('/')[-2]} n={n} warm total {tot:.2f} ms")
for k,v in sorted(agg.items(), key=lambda kv:-kv[1]):
    if v > 0.002: print(f"  {k:<40} {v:8.3f} ms {100*v/tot:5.1f}%")
print("  -- every Subgraph node --")
for x in sorted(info, key=lambda x:-x.real_time.total_seconds()):
    if x.node_type == 'Subgraph':
        print(f"    {x.real_time.total_seconds()*1e3:8.3f}  {x.node_name}")
