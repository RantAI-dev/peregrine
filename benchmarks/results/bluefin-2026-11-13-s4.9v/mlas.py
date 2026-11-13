#!/usr/bin/env python3
"""MLAS's rate on the same float GEMM shapes and row counts, through ONNX Runtime.

One MatMul node per graph: activation [M, K] × weight [K, N] with the weight a graph
initializer, so ONNX Runtime packs it once — the state Peregrine's `fpack` is in too
(the panels are built at load). `intra_op_num_threads` is pinned, the call is warmed up,
and the reported rate is GMAC/s of the median of five timed calls. The model is written
as a protobuf here because this host's Python has ONNX Runtime but not the `onnx` package.

  taskset -c 0-3 /var/tmp/embed-venv/bin/python \\
      benchmarks/results/bluefin-2026-11-13-s4.9v/mlas.py 4
"""
import json
import os
import statistics
import struct
import sys
import time

import numpy as np
import onnxruntime as ort

SHAPES = [(384, 384), (384, 1536), (768, 768), (768, 3072), (384, 3072), (768, 1536)]
ROWS = [16, 32, 64, 96, 128, 192, 256, 512]


def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def tag(field, wire):
    return varint((field << 3) | wire)


def ld(field, payload):
    return tag(field, 2) + varint(len(payload)) + payload


def vi(field, value):
    return tag(field, 0) + varint(value)


def dim(d):
    if isinstance(d, int):
        return ld(1, vi(1, d))
    return ld(1, ld(2, d.encode()))


def value_info(name, dims):
    tensor = vi(1, 1)  # elem_type = FLOAT
    shape = b"".join(dim(d) for d in dims)
    tensor += ld(2, shape)
    return ld(1, name.encode()) + ld(2, ld(1, tensor))


def model_bytes(n_in, n_out, weight):
    node = ld(1, b"x") + ld(1, b"W") + ld(2, b"y") + ld(4, b"MatMul")
    tensor = b"".join(vi(1, d) for d in (n_in, n_out))
    tensor += vi(2, 1)  # FLOAT
    tensor += ld(9, weight.astype("<f4").tobytes())  # raw_data
    tensor += ld(8, b"W")
    graph = ld(1, node)
    graph += ld(2, b"gemm")
    graph += ld(5, tensor)
    graph += ld(11, value_info("x", ["m", n_in]))
    graph += ld(12, value_info("y", ["m", n_out]))
    model = vi(1, 8)  # ir_version
    model += ld(7, graph)
    model += ld(8, vi(2, 17))  # opset_import
    return model


def session(n_in, n_out, threads):
    # The weight's values are irrelevant to the schedule; identity-like noise keeps
    # the packer's work identical to a real matrix's.
    w = np.random.default_rng(11).standard_normal((n_in, n_out), dtype=np.float32)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_bytes(n_in, n_out, w), so,
                                providers=["CPUExecutionProvider"])


def rate(sess, n_in, n_out, m, reps=5):
    x = np.random.default_rng(7).standard_normal((m, n_in), dtype=np.float32)
    for _ in range(3):
        sess.run(None, {"x": x})
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sess.run(None, {"x": x})
        times.append(time.perf_counter() - t0)
    secs = statistics.median(times)
    return (m * n_in * n_out) / secs / 1e9, secs


def main():
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out = {"engine": "onnxruntime", "version": ort.__version__, "threads": threads, "cells": {}}
    print(f"onnxruntime {ort.__version__}, {threads} thread(s), MLAS MatMul, GMAC/s")
    for n_in, n_out in SHAPES:
        sess = session(n_in, n_out, threads)
        print(f"\n== n_in {n_in}, n_out {n_out}")
        print(f"{'rows':>5} {'GMAC/s':>9} {'ms':>9}")
        for m in ROWS:
            g, s = rate(sess, n_in, n_out, m)
            out["cells"][f"{n_in}x{n_out}_m{m}"] = {"gmac_s": g, "ms": s * 1e3}
            print(f"{m:>5} {g:>9.1f} {s * 1e3:>9.3f}")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"mlas-{threads}t.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nwrote {os.path.basename(path)}")


if __name__ == "__main__":
    main()
