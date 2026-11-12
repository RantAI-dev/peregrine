#!/usr/bin/env python3
"""OpenVINO's own per-layer times for all-MiniLM-L6-v2 and bge-small, 4 threads.

Reads the same ONNX the embeddings harness gives the OpenVINO backend, compiles it
for CPU with profiling on, runs one 512-token inference, and aggregates the
per-node `real_time` by node type.
"""
import sys
from collections import defaultdict

import numpy as np
import openvino as ov
import openvino.properties as props

path, n = sys.argv[1], int(sys.argv[2])

core = ov.Core()
model = core.read_model(path)
model.reshape({i.get_any_name(): ov.PartialShape([1, n]) for i in model.inputs})
cfg = {"INFERENCE_NUM_THREADS": 4, "PERF_COUNT": True}
compiled = core.compile_model(model, "CPU", cfg)

ids = np.zeros((1, n), dtype=np.int64)
mask = np.ones((1, n), dtype=np.int64)
request = compiled.create_infer_request()
inputs = {i.get_any_name(): v for i, v in zip(model.inputs, [ids, mask, ids])}
request.infer(inputs)
info = request.get_profiling_info()

by_type = defaultdict(float)
count = defaultdict(int)
for node in info:
    by_type[node.node_type] += node.real_time.total_seconds() * 1e3
    count[node.node_type] += 1
total = sum(by_type.values())
print(f"{path} n={n} 4 threads: openvino per-node real time, {len(info)} nodes")
for t, ms in sorted(by_type.items(), key=lambda kv: -kv[1]):
    if ms >= 0.5:
        print(f"  {t:<24} {ms:8.2f} ms  {100*ms/total:5.1f}%  ({count[t]} nodes)")
print(f"  {'TOTAL':<24} {total:8.2f} ms")
