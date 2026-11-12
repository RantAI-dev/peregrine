#!/usr/bin/env python3
"""The fork's `q8_k_r16` codes and scales, from its own source lines, for the rows
Peregrine converted (S5.4a conformance).

`iqk_convert_q4_k_r4_q8_k_r16` (ik_llama.cpp `479074c`,
`ggml/src/iqk/iqk_gemm_kquants.cpp:2400`) and `iqk_convert_q6_k_r4_q8_k_r16`
(`:2724`) are the conversions `docs/design/s5.4-fast-prompt.md` §2 describes. This
script repeats their arithmetic on the rows `<prefix>-rows.bin` holds and writes
`<prefix>-python.bin` in the same canonical layout `examples/kq8model.rs` writes, so
`cmp` is the comparison of every code and every scale.

Every step is the fork's, including the two that §2's wording gets slightly wrong:

  * the f32 weights of Q4_K are one fused multiply-add of the min term —
    `fma(q·Ld as f32, d, (−dmin)·Lm)` (`:2444-2463`) — and Q6_K never forms a
    weight at all: its products are the int16 `(q − 32)·L`;
  * `dq = max|w| / 127` and the codes are `round_to_nearest_even(w · (1/dq))`, a
    rounded reciprocal rather than a division (`:2464`, `PackerQ80::convert`, `:16-26`);
  * the stored byte is the code biased by +128 (the `xor 0x80` of `:2471-2474`);
  * Q6_K's exact case is `max|p| · (1/127) < 1`, and then the codes are the products
    and the scale is `d` (`iqk_common.h:638-687`).

`fmaf` (glibc's correctly-rounded f32 fused multiply-add) and `struct`'s f16/f32
round trips give the same single-rounding arithmetic the fork's SSE gets, so the
comparison is of the codes and scales and not of a re-derived approximation.

    python3 fork_requant.py <prefix>
"""

import ctypes
import math
import struct
import sys

libm = ctypes.CDLL("libm.so.6")
libm.fmaf.restype = ctypes.c_float
libm.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]


def f32(x):
    """Round a Python float to the nearest f32."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f16(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def fmaf(a, b, c):
    return libm.fmaf(a, b, c)


def get_scale_min_k4(j, q):
    """ggml's `get_scale_min_k4` (`ggml-quants.c:2111`)."""
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4))


def requant_f32(w):
    """`dq = max|w|/127`, `id = 1/dq`, `round_to_nearest_even(w·id)`."""
    top = max(abs(x) for x in w)
    dq = f32(top / 127.0)
    idq = f32(1.0 / dq) if dq != 0.0 else 0.0
    return dq, [round(f32(x * idq)) for x in w]


def requant_q6k(products, d):
    """The fork's Q6_K branch: exact when `max|p|·(1/127) < 1`, rounded otherwise."""
    top = max(abs(x) for x in products)
    dnew = f32(f32(top) * f32(1.0 / 127.0))
    if dnew < 1.0:
        return d, list(products)
    idq = f32(1.0 / dnew) if abs(dnew) > 1e-9 else 0.0
    return f32(d * dnew), [round(f32(x * idq)) for x in products]


def q4k_row(block):
    """The 256 f32 weights of one Q4_K block, the fork's `:2444-2463`."""
    d = f16(struct.unpack("<H", block[0:2])[0])
    dmin = f16(struct.unpack("<H", block[2:4])[0])
    scales = block[4:16]
    qs = block[16:144]
    w = []
    for j in range(8):
        ld, lm = get_scale_min_k4(j, scales)
        mm = f32((-dmin) * lm)
        run = qs[(j // 2) * 32:(j // 2) * 32 + 32]
        for i in range(32):
            q = (run[i] & 0x0F) if j % 2 == 0 else (run[i] >> 4)
            w.append(fmaf(f32(q * ld), d, mm))
    return w


def q6k_products(block):
    """The 256 int products `(q − 32)·L` of one Q6_K block (`convert_to_q8_k_r8`).

    The six-bit codes are `convert_q6_k`'s (`iqk_quantize.cpp:7387`): within a
    128-weight half, `L[n+l]`, `L[n+l+32]`, `L[n+l+64]` and `L[n+l+96]` take their
    low nibbles from `ql[l]` and `ql[l+32]` and their high bits from bits `0`/`2`
    and `4`/`6` of `qh[l]`.
    """
    ql = block[0:128]
    qh = block[128:192]
    sc = block[192:208]
    p = []
    for weight in range(256):
        n = weight // 128
        i = weight % 128
        l = i % 32
        k = i // 32
        if k == 0:
            nib = ql[n * 64 + l] & 0xF
        elif k == 1:
            nib = ql[n * 64 + l + 32] & 0xF
        elif k == 2:
            nib = ql[n * 64 + l] >> 4
        else:
            nib = ql[n * 64 + l + 32] >> 4
        code = nib | (((qh[n * 32 + l] >> (2 * k)) & 3) << 4)
        scale = struct.unpack("<b", sc[weight // 16:weight // 16 + 1])[0]
        p.append((code - 32) * scale)
    return p


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "rows"
    with open(prefix + "-rows.bin", "rb") as f:
        data = f.read()
    if data[:4] != b"KQ8R":
        raise SystemExit("not a rows file")
    fmt, n_sb, n_groups, row_bytes = struct.unpack("<IIII", data[4:20])
    block = 144 if fmt == 0 else 210
    assert row_bytes == n_sb * block, (row_bytes, n_sb, block)
    rows = data[20:]
    weights = 0
    with open(prefix + "-python.bin", "wb") as o:
        o.write(b"KQ8C")
        o.write(struct.pack("<IIII", 1, fmt, n_sb, n_groups))
        for g in range(n_groups):
            for k in range(16):
                base = (g * 16 + k) * row_bytes
                for sb in range(n_sb):
                    blk = rows[base + sb * block:base + (sb + 1) * block]
                    if fmt == 0:
                        d, codes = requant_f32(q4k_row(blk))
                    else:
                        d = f16(struct.unpack("<H", blk[208:210])[0])
                        d, codes = requant_q6k(q6k_products(blk), d)
                    o.write(struct.pack("<f", d))
                    o.write(bytes((c & 0xFF) for c in codes))
                    weights += 256
    print(f"{prefix}-python.bin: {n_groups} groups of 16 rows, {n_sb} super-blocks, "
          f"{weights} weights by the fork's arithmetic")


main()
