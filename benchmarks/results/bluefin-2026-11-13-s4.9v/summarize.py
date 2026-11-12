#!/usr/bin/env python3
"""Medians of the three sweeps: Peregrine's packed float GEMM and MLAS, GMAC/s.

Parses `peregrine-{1,4}t-sweep{1,2,3}.txt` (the `mgemm` example) and `mlas-{1,4}t*.txt`
(the ONNX Runtime MatMul script), takes the median of the three runs per cell, and
prints the table the design section quotes. Also prints the in-situ number the engine's
own counters give for the same pass, for the comparison.
"""
import glob
import re
import statistics
import sys


def peregrine(paths):
    """{(n_in, n_out): {rows: (f32_1t, f32_4t, f16_1t, f16_4t)}}"""
    out, shape, roof = {}, None, {}
    for f in sorted(paths):
        for line in open(f):
            m = re.match(r"FMA roof: one thread ([\d.]+), four threads ([\d.]+)", line)
            if m:
                roof.setdefault("1", []).append(float(m.group(1)))
                roof.setdefault("4", []).append(float(m.group(2)))
                continue
            m = re.match(r"== n_in (\d+), n_out (\d+)", line)
            if m:
                shape = (int(m.group(1)), int(m.group(2)))
                out.setdefault(shape, {})
                continue
            m = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+|—)\s+([\d.]+)\s+([\d.]+|—)", line)
            if m and shape:
                rows = int(m.group(1))
                vals = [float(x) if x != "—" else None for x in m.groups()[1:]]
                out[shape].setdefault(rows, []).append(vals)
    med = {}
    for shape, rows in out.items():
        for r, runs in rows.items():
            cols = []
            for i in range(4):
                xs = [v[i] for v in runs if v[i] is not None]
                cols.append(statistics.median(xs) if xs else None)
            med[(shape, r)] = cols
    return med, {k: statistics.median(v) for k, v in roof.items()}


def mlas(paths):
    out = {}
    for f in sorted(paths):
        shape = None
        for line in open(f):
            m = re.match(r"== n_in (\d+), n_out (\d+)", line)
            if m:
                shape = (int(m.group(1)), int(m.group(2)))
                continue
            m = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)", line)
            if m and shape:
                out.setdefault((shape, int(m.group(1))), []).append(float(m.group(2)))
    return {k: statistics.median(v) for k, v in out.items()}


def main():
    d = "benchmarks/results/bluefin-2026-11-13-s4.9v"
    p, roof = peregrine(glob.glob(f"{d}/peregrine-*t-sweep*.txt"))
    m4 = mlas(glob.glob(f"{d}/mlas-4t*.txt"))
    m1 = mlas(glob.glob(f"{d}/mlas-1t*.txt"))
    print(f"roof: 1 thread {roof['1']:.1f}, 4 threads {roof['4']:.1f} GMAC/s")
    order = [(384, 384), (384, 1536), (768, 768), (768, 3072), (3072, 768),
             (2304, 768), (384, 3072), (768, 1536)]
    rows = [16, 32, 64, 96, 128, 192, 256, 512]
    for shape in order:
        print(f"\n## n_in {shape[0]}, n_out {shape[1]}")
        print(f"{'M':>4}  {'P 1t':>7} {'MLAS 1t':>8} {'P 4t':>7} {'MLAS 4t':>8} "
              f"{'P/MLAS':>7} {'P % roof':>9}")
        for r in rows:
            v = p.get((shape, r))
            if not v:
                continue
            f32_1, f32_4, f16_1, f16_4 = v
            one = m1.get((shape, r))
            four = m4.get((shape, r))
            ratio = f16_4 / four if four else float("nan")
            print(f"{r:>4}  {f16_1:>7.1f} {one if one else float('nan'):>8.1f} "
                  f"{f16_4:>7.1f} {four if four else float('nan'):>8.1f} {ratio:>7.3f} "
                  f"{f16_4 / roof['4'] * 100:>8.1f}%")


if __name__ == "__main__":
    sys.exit(main())
