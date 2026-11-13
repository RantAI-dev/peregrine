#!/usr/bin/env python3
"""Write one arm of the S4.9w row-block three-arm measurement into linear.rs.

Local builds only: run it, `cargo build --release -p peregrine`, copy the binary, then
`git checkout -- crates/peregrine-core/src/blocks/linear.rs`.  The three arms differ in
nothing but `fp_row_block` (old cap, whole pass, 2 MiB bound); 1m/4m move the constant.

  python3 part1-variant.py old|whole|bound|1m|4m [repo-root]
"""
import sys
from pathlib import Path

arm = sys.argv[1]
root = Path(sys.argv[2] if len(sys.argv) > 2 else __file__).resolve()
if root.is_file():
    root = root.parent.parent.parent
p = root / "crates/peregrine-core/src/blocks/linear.rs"
src = p.read_text()

body = """    let fit = (FP_BLOCK_BYTES / (4 * n_in.max(1))).max(peregrine_kernels::fpack::row_block(n_in));
    nb.div_ceil(want).min(fit).max(1)"""

if arm == "old":
    new = """    nb.div_ceil(want)
        .min(peregrine_kernels::fpack::row_block(n_in))
        .max(1)"""
    assert body in src, "bound body not found"
    src = src.replace(body, new)
elif arm == "whole":
    new = "    nb.div_ceil(want).max(1)"
    assert body in src, "bound body not found"
    src = src.replace(body, new)
elif arm in ("1m", "4m"):
    n = {"1m": "1", "4m": "4"}[arm]
    old = "const FP_BLOCK_BYTES: usize = 2 << 20;"
    assert old in src, "const not found"
    src = src.replace(old, f"const FP_BLOCK_BYTES: usize = {n} << 20;")
elif arm != "bound":
    raise SystemExit(f"unknown arm {arm}")

p.write_text(src)
print(f"wrote {arm} into {p}")
