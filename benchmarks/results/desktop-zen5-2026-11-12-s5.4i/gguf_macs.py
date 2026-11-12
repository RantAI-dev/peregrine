#!/usr/bin/env python3
"""The MAC counts of one prompt row, from a GGUF file's own tensor table.

    gguf_macs.py <model.gguf> [<model.gguf> ...]

Prints one JSON object per file: the tensor table grouped into the engine's matrix groups
(`qkv`, `wo`, `gate/up`, `down`, `head`), the MACs of one activation row through each, the
same split by weight format, the bytes of weights in each format, and the config numbers the
attention arithmetic needs. `account.py` multiplies these by the rows a pass ran.

An output row costs `dims[0]` MACs per activation row (one per weight of the row), so a
group's MACs per row are `Σ dims[0] · dims[1]` over its matrices.
"""

import json
import struct
import sys

FMT = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    30: "BF16",
}

# Bytes per 256 weights (GGML block sizes), for the formation's byte accounting.
BLOCK = {
    "Q4_K": (144, 256),
    "Q6_K": (210, 256),
    "Q5_K": (176, 256),
    "Q5_0": (22, 32),
    "Q8_0": (34, 32),
    "F32": (4, 1),
    "F16": (2, 1),
    "BF16": (2, 1),
}

GROUP = {
    "attn_q": "qkv",
    "attn_k": "qkv",
    "attn_v": "qkv",
    "attn_qkv": "qkv",
    "attn_output": "wo",
    "ffn_gate": "gate/up",
    "ffn_up": "gate/up",
    "ffn_gate_up": "gate/up",
    "ffn_down": "down",
    "output": "head",
}


def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8")


def read_value(f, t):
    sizes = {
        0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
        6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
    }
    if t in sizes:
        s = struct.calcsize(sizes[t])
        return struct.unpack(sizes[t], f.read(s))[0]
    if t == 8:
        return read_string(f)
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [read_value(f, et) for _ in range(n)]
    raise ValueError(f"GGUF value type {t}")


def tensor_group(name):
    parts = name.split(".")
    tail = parts[-2] if parts[-1] == "weight" and len(parts) > 1 else parts[-1]
    if name == "output.weight":
        return "head"
    if name == "token_embd.weight" or name.endswith("token_embd.weight"):
        return "embed"
    for k, g in GROUP.items():
        if tail == k:
            return g
    return "other"


def parse(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        version = struct.unpack("<I", f.read(4))[0]
        assert version >= 2, version
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        meta = {}
        for _ in range(n_kv):
            key = read_string(f)
            t = struct.unpack("<I", f.read(4))[0]
            meta[key] = read_value(f, t)
        # The tensor table follows the metadata with no padding (the alignment the
        # format names applies to the tensor *data*).
        tensors = []
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack("<I", f.read(4))[0]
            _off = struct.unpack("<Q", f.read(8))[0]
            tensors.append((name, dims, ttype))

    groups = {}
    formats = {}
    fbytes = {}
    for name, dims, ttype in tensors:
        fmt = FMT.get(ttype, str(ttype))
        g = tensor_group(name)
        # An embedding row is not multiplied in a prompt pass (it is a gather); the
        # matrices are the ones with an output dimension.
        if g in ("embed",):
            continue
        if len(dims) != 2:
            continue
        macs = dims[0] * dims[1]
        groups[g] = groups.get(g, 0) + macs
        if g == "head":
            # The head is multiplied for the rows that want logits — one row of a
            # prompt pass — so its weights are not part of the per-row format total.
            continue
        formats[fmt] = formats.get(fmt, 0) + macs
        b, w = BLOCK.get(fmt, (0, 1))
        fbytes[fmt] = fbytes.get(fmt, 0) + macs // w * b

    arch = meta.get("general.architecture", "?")
    cfg = {
        "path": path,
        "arch": arch,
        "d_model": meta.get(f"{arch}.embedding_length"),
        "n_layer": meta.get(f"{arch}.block_count"),
        "n_head": meta.get(f"{arch}.attention.head_count"),
        "n_kv_head": meta.get(f"{arch}.attention.head_count_kv"),
        "head_dim": None,
        "vocab": None,
    }
    kl = meta.get(f"{arch}.attention.key_length")
    vl = meta.get(f"{arch}.attention.value_length")
    if kl:
        cfg["head_dim"] = kl
    elif cfg["d_model"] and cfg["n_head"]:
        cfg["head_dim"] = cfg["d_model"] // cfg["n_head"]
    cfg["vocab"] = meta.get(f"{arch}.vocab_size") or len(
        [k for k in meta if k.startswith("tokenizer.ggml.tokens")]
    )
    # The head's own row count, from the output tensor.
    for name, dims, ttype in tensors:
        if name == "output.weight":
            cfg["vocab"] = dims[1]
            cfg["head_macs"] = dims[0] * dims[1]
    if "head_macs" not in cfg and cfg["d_model"] and cfg["vocab"]:
        cfg["head_macs"] = cfg["d_model"] * cfg["vocab"]
    return {
        "cfg": cfg,
        "group_macs": groups,
        "format_macs": formats,
        "format_bytes": fbytes,
        "n_tensors": len(tensors),
    }


if __name__ == "__main__":
    out = {}
    for p in sys.argv[1:]:
        r = parse(p)
        out[r["cfg"]["arch"] + ":" + p.split("/")[-1]] = r
    print(json.dumps(out, indent=1))
