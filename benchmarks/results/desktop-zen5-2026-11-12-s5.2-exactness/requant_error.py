#!/usr/bin/env python3
"""The fork's requantisation error, sampled from a real model's Q4_K weights.

`iqk_convert_q4_k_r4_q8_k_r16` (ik_llama.cpp `479074c`,
`ggml/src/iqk/iqk_gemm_kquants.cpp:2400`) dequantises each 256-weight row-block to f32, takes
its largest magnitude (`:2455`) and rounds every weight onto a grid of `max/127` (`:2464`).
This script does exactly that to the Q4_K tensors of a GGUF file and reports the error, so the
design note's `max/254` bound is checked against real weights rather than only derived.

No GGUF metadata is read past the tensor table, and nothing is written to the model's
directory. Only Q4_K tensors (type 12) are touched, and every `stride`-th super-block is
sampled: a 1B model has ~1.4 million of them and the sample is stated in the output.

    python3 requant_error.py <model.gguf> [stride]
"""

import math
import struct
import sys

# ggml's type ids, from ggml.h.
Q4_K = 12
# The metadata value types that can appear in a KV record.
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n)


def skip_value(f, t):
    if t in SCALAR:
        f.read(SCALAR[t])
    elif t == 8:
        read_str(f)
    elif t == 9:
        et, n = struct.unpack("<IQ", f.read(12))
        if et in SCALAR:
            f.read(n * SCALAR[et])
        elif et == 8:
            for _ in range(n):
                read_str(f)
        else:
            raise SystemExit(f"array of type {et} is not a GGUF value type")
    else:
        raise SystemExit(f"metadata value type {t} is not a GGUF value type")


def load(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit("not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version != 3:
            raise SystemExit(f"GGUF version {version}, expected 3")
        alignment = 32
        for _ in range(n_kv):
            key = read_str(f).decode("utf-8", "replace")
            (t,) = struct.unpack("<I", f.read(4))
            if key == "general.alignment":
                if t != 4:
                    raise SystemExit("general.alignment is not u32")
                (alignment,) = struct.unpack("<I", f.read(4))
            else:
                skip_value(f, t)
        infos = []
        for _ in range(n_tensors):
            name = read_str(f).decode("utf-8", "replace")
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            t, off = struct.unpack("<IQ", f.read(12))
            infos.append((name, dims, t, off))
        pos = f.tell()
        return infos, (pos + alignment - 1) // alignment * alignment


def six_bit(s):
    """The eight 6-bit scales and the eight 6-bit mins of one super-block."""
    out = []
    for j in range(8):
        if j < 4:
            sc, m = s[j] & 0x3F, s[j + 4] & 0x3F
        else:
            sc = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4)
            m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4)
        out.append((sc, m))
    return out


def main():
    path = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    infos, base = load(path)
    blocks = 0
    weights = 0
    worst = 0.0
    sum_ratio = 0.0
    sum_sq = 0.0
    lo, hi = 1.0, 0.0
    exact = 0
    seen = 0
    with open(path, "rb") as f:
        for name, dims, t, off in infos:
            if t != Q4_K:
                continue
            ne0 = dims[0]
            rows = 1
            for d in dims[1:]:
                rows *= d
            n_sb = ne0 // 256
            f.seek(base + off)
            raw = f.read(rows * n_sb * 144)
            for b in range(0, rows * n_sb, stride):
                a = raw[b * 144 : (b + 1) * 144]
                d = struct.unpack("<e", a[0:2])[0]
                dmin = struct.unpack("<e", a[2:4])[0]
                scales = six_bit(a[4:16])
                qs = a[16:144]
                blocks += 1
                w = []
                for j, (sc, m) in enumerate(scales):
                    base_ = d * sc
                    mn = dmin * m
                    # Each of the 128 nibble bytes holds two codes: the low
                    # nibbles are sub-block 2·(byte/32) and the high nibbles
                    # 2·(byte/32)+1, thirty-two weights each.
                    run = qs[(j // 2) * 32 : (j // 2) * 32 + 32]
                    for i in range(32):
                        byte = run[i]
                        q = (byte & 0x0F) if j % 2 == 0 else (byte >> 4)
                        w.append(base_ * q - mn)
                top = max(abs(x) for x in w)
                if top == 0.0:
                    continue
                dq = top / 127.0
                err_sum = 0.0
                for x in w:
                    c = round(x / dq)
                    e = abs(x - c * dq)
                    if e == 0.0:
                        exact += 1
                    err_sum += e
                    if e > worst * top:
                        worst = e / top
                r = err_sum / len(w) / top
                sum_ratio += r
                sum_sq += r * r
                lo = min(lo, r)
                hi = max(hi, r)
                weights += len(w)
    print(f"model {path}")
    print(f"Q4_K super-blocks sampled: {blocks} (every {stride}th), weights {weights}")
    print(f"mean of the per-block mean |error| / block max: {sum_ratio/blocks:.5f}")
    print(f"that ratio across blocks: {lo:.5f} .. {hi:.5f}")
    print(f"worst single weight / block max: {worst:.5f}   (bound 0.5/127 = {0.5/127:.5f})")
    print(f"weights exactly on the grid: {100.0*exact/weights:.2f} %")
    print()
    print("The grid step is max/127, so the bound is half a step, max/254 = 0.3937 %; 'block max'")
    print("is the largest |w| in that row's 256-weight super-block, the number the fork divides")
    print("by 127. A weight of average size is off by more than this fraction of *itself*: the")
    print("figures above are relative to the block's maximum, as the design note's bound is.")


main()
