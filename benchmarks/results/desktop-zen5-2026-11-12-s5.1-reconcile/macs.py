#!/usr/bin/env python3
"""Multiply-accumulates per prompt token, read from the model file's own tensor shapes.

Only the tensor directory and its shapes are read; no metadata value is printed except
integer/float hyper-parameters, and no chat template is touched.

Per token, one matmul contributes K*M MACs where the weight is [M out, K in]. The
per-layer trunk is q, k, v, o, gate, up, down; the token embedding is a row gather (no
MACs) and the output head runs for the last token only, so it is excluded (it is +1/512
of a 512-token pass, which is reported separately).

    python3 macs.py <file.gguf> [...]
"""
import struct
import sys

TYPES = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32", 7: "bool",
         8: "string", 9: "array", 10: "u64", 11: "i64", 12: "f64"}
SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
          6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


def read_string(f):
    n, = struct.unpack("<Q", f.read(8))
    return f.read(n)


def skip_value(f, t):
    if t == 8:
        read_string(f)
    elif t == 9:
        et, = struct.unpack("<I", f.read(4))
        n, = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            skip_value(f, et)
    else:
        fmt, size = SCALAR[t]
        f.read(size)


def gguf(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        version, = struct.unpack("<I", f.read(4))
        n_tensors, = struct.unpack("<Q", f.read(8))
        n_kv, = struct.unpack("<Q", f.read(8))
        for _ in range(n_kv):
            read_string(f)
            t, = struct.unpack("<I", f.read(4))
            skip_value(f, t)
        out = []
        for _ in range(n_tensors):
            name = read_string(f).decode()
            nd, = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            ty, = struct.unpack("<I", f.read(4))
            f.read(8)  # offset
            out.append((name, dims, ty))
        return version, out


def main():
    for path in sys.argv[1:]:
        version, tensors = gguf(path)
        trunk = {}
        for name, dims, _ty in tensors:
            if ".attn_" in name or ".ffn_" in name:
                kind = name.split(".")[2].replace(".weight", "")
                k, m = dims[0], (dims[1] if len(dims) > 1 else 1)
                trunk[kind] = trunk.get(kind, 0) + k * m
        n_layer = len({name.split(".")[1] for name, _, _ in tensors if name.startswith("blk.")})
        head = sum(k * m for name, d, _ in tensors
                   if name == "output.weight" for k, m in [(d[0], d[1] if len(d) > 1 else 1)])
        per_layer = sum(trunk.values()) / n_layer
        total = per_layer * n_layer
        print(f"{path.split('/')[-1]}  (GGUF v{version}, {n_layer} layers)")
        for kind in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
            print(f"  {kind:12s} {trunk.get(kind, 0)/1e6:9.3f} MMAC per layer")
        print(f"  trunk        {total/1e9:.4f} GMAC per token")
        if head:
            print(f"  output head  {head/1e9:.4f} GMAC (last token only; "
                  f"+{(head/total)/512*100:.2f} % of a 512-token pass)")
        print()


if __name__ == "__main__":
    main()
