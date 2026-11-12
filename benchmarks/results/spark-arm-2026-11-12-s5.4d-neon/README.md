# NEON bodies for the 8-bit prompt kernel (S5.4d, ARM)

S5.4a defined the 8-bit prompt form and wrote its scalar twin, its AVX-512 bodies
and the `kqbench` rows on the desktop. This round writes the aarch64 bodies: the
build of the Q4_K, Q6_K and Q5_0 forms and the multiply on `sdot`
(`neon-dotprod`) and on `smmla` (`neon-i8mm`), all bit-identical to the twin. It
also measures the resident form against the exact tile kernels on the shapes the
S5.4a table used. Nothing is dispatched in the engine yet: the option arrives
with S5.4b on x86, and no default-path file changed.

- Date: 2026-11-12, 14:40–15:10 local (UTC+7).
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores in two clusters:
  10 × Cortex-X925 at 3.9 GHz (CPUs 5–9 and 15–19) and 10 × Cortex-A725 at
  2.8 GHz, 121 GB unified memory, Linux `6.17.0-1008-nvidia`, governor
  `performance`, rustc 1.99.1.
- Peregrine: this branch on `origin/main` at `f15594b` (S5.4a's #232), a clean
  worktree, `cargo build --release -p peregrine-bench --example neonbench`.
- A vLLM co-tenant was on the host for the whole window (1-minute load average
  6.2–7.8, `MemAvailable` 35–36 GB of 121); it left at 15:06, after the runs.
  Every figure below is a median of three runs, and the ten-thread figure is ten
  one-thread processes pinned one per Cortex-X925 core, summed per line.

## What was built

`crates/peregrine-kernels/src/neon.rs`:

- `q8k_build_q4k_neon`, `q8k_build_q6k_neon`, `q8k_build_q5_0_neon`: the build
  vectorised with NEON. A 32-weight sub-block is four eight-byte nibble loads
  (Q4_K) or two 8-byte nibble/high-bit loads per 16 (Q6_K); the codes widen to
  f32, Q4_K's weight is one fused multiply-add of the exact `q·Ld` integer
  product with `(−dmin)·Lm`, and Q5_0's is the format's `d·(q−16)`. The
  requantisation takes the vector max, `dq = m/127` and `id = 1/dq` in scalar
  (the twin's two divisions), then `vcvtnq_s32_f32` (round to nearest, ties to
  even), a saturating `vqmovn` pair, and the form's `+128` bias.
- `q8k_gemm_sdot`: eight rows per super-block, 32 groups of eight codes; each
  16-byte load is two output rows, the activation's eight codes are duplicated
  into the second operand, and every group accumulates into four int32 row-pair
  lanes. The bias is undone with one `eor 0x80` per load, so the integer sum is
  `Σ code·a` directly and the epilogue is the twin's `d·dk` then one fused
  multiply-add, super-block by super-block.
- `q8k_gemm_i8mm`: the same form on `smmla`. Two output rows × two activations
  per instruction; four row pairs and two activation pairs cover the eight rows
  and four activation rows of a group, and the 1–3-row tail falls back to the
  `sdot` body.
- `q8k_gemm`: the tier entry point — `smmla` on `neon-i8mm`, `sdot` on
  `neon-dotprod`, the scalar twin otherwise. It is not called by the engine.

The bias: the form stores `code + 128` for x86's unsigned `vpdpbusd`. `sdot` and
`smmla` are signed, so this file flips bit 7 (`eor 0x80`) and uses the codes as
stored minus the bias. The float epilogue is the twin's expression in its order:
the integer sums are exact in any grouping, the bias is exact in int32, and the
fold is one `d·dk` multiply and one `fma` per (row, super-block), block by block.

## Conformance

`cargo test --release -p peregrine-kernels` has three new tests:

- the NEON build against the twin, byte for byte, on a random Q4_K tile (1 and 3
  super-blocks), the all-zero tile, the one-huge-weight tile, the six-bit limits
  tile and the exact-half tile; on Q6_K's exact case, one just outside it, and
  random tiles; and on Q5_0 tiles of 8, 16 and 21 blocks (the last ends in a short
  super-block, which the build zero-fills as the twin does);
- both multiply bodies against `kq8::dot_q8k` bit for bit on four forms (random
  Q4_K, the tie tile, the just-outside Q6_K block, a Q5_0 form) at every width
  1..=40 and 63, 64, 65, 127, 128, 129, 511, 512, 513 activation rows;
- the tier entry point: under `PEREGRINE_ISA=neon-i8mm` it calls the `smmla`
  body (counter), under `PEREGRINE_ISA=neon-dotprod` the `sdot` body, and the
  `neon` and `scalar` caps run the scalar twin.

`cargo test --release -p peregrine-kernels -p peregrine-core -p peregrine-models`
passes natively and under `PEREGRINE_ISA=scalar`, `neon`, `neon-dotprod` and
`neon-i8mm`. No golden changed.

## Kernel speed

`neonbench` (one thread pinned to core 5, or ten copies pinned one per
Cortex-X925 core), GMAC/s counting `8 output rows · n weights · nb activations`
per tile call. `rows` is the engine shape, so `rows/8` tiles (capped at 256) are
walked per iteration: at `8192 2048` that is 256 forms (17 MB of Q4_K form, plus
the Q6_K/Q5_0 forms) streamed from L3/DRAM every iteration; at `4096 512` the 64
tiles are L2/L3-resident. Reproduce:

```sh
cargo build --release -p peregrine-bench --example neonbench
taskset -c 5 target/release/examples/neonbench 4096 512 3
taskset -c 5 target/release/examples/neonbench 8192 2048 3
# ten threads: ten copies, one per core 5,6,7,8,9,15,16,17,18,19, summed per line
```

The raw runs are `neonbench-4096x512-1t.txt`, `-10t.txt`,
`neonbench-8192x2048-1t.txt` and `-10t.txt`.

### 4096 × 512: 4 096 weights × 512 rows (the S4.9q layer shape)

**sdot tier**

| Format | Rows | exact 1T | form 1T | ratio | exact 10T | form 10T | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 1 | 50.0 | 59.8 | 1.20× | 396.4 | 351.0 | 0.89× |
| Q4_K | 8 | 75.7 | 169.7 | 2.24× | 715.1 | 1483.7 | 2.07× |
| Q4_K | 16 | 76.2 | 176.1 | 2.31× | 749.8 | 1711.9 | 2.28× |
| Q4_K | 64 | 77.2 | 180.0 | 2.33× | 740.3 | 1761.9 | 2.38× |
| Q4_K | 128 | 77.2 | 182.0 | 2.36× | 738.7 | 1776.3 | 2.40× |
| Q4_K | 512 | 76.3 | 176.2 | 2.31× | 728.6 | 1732.5 | 2.38× |
| Q6_K | 1 | 33.4 | 63.8 | 1.91× | 271.7 | 389.3 | 1.43× |
| Q6_K | 8 | 51.3 | 172.8 | 3.37× | 500.5 | 1581.7 | 3.16× |
| Q6_K | 16 | 51.8 | 176.2 | 3.40× | 515.3 | 1722.3 | 3.34× |
| Q6_K | 64 | 52.1 | 179.3 | 3.44× | 502.1 | 1763.5 | 3.51× |
| Q6_K | 128 | 52.2 | 181.9 | 3.48× | 501.2 | 1774.8 | 3.54× |
| Q6_K | 512 | 52.1 | 176.6 | 3.39× | 495.9 | 1728.2 | 3.48× |
| Q5_0 | 1 | 21.6 | 44.7 | 2.07× | 198.7 | 373.3 | 1.88× |
| Q5_0 | 8 | 35.2 | 174.0 | 4.94× | 320.3 | 1566.2 | 4.89× |
| Q5_0 | 16 | 35.2 | 177.1 | 5.03× | 315.7 | 1691.1 | 5.36× |
| Q5_0 | 64 | 35.2 | 180.2 | 5.12× | 314.8 | 1774.8 | 5.64× |
| Q5_0 | 128 | 35.3 | 181.5 | 5.14× | 314.6 | 1783.5 | 5.67× |
| Q5_0 | 512 | 34.8 | 177.3 | 5.09× | 310.0 | 1731.5 | 5.59× |

**i8mm tier**

| Format | Rows | exact 1T | form 1T | ratio | exact 10T | form 10T | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 1 | 48.6 | 65.5 | 1.35× | 376.3 | 381.9 | 1.01× |
| Q4_K | 8 | 121.6 | 267.1 | 2.20× | 1155.2 | 2382.3 | 2.06× |
| Q4_K | 16 | 122.7 | 286.1 | 2.33× | 1186.3 | 2667.0 | 2.25× |
| Q4_K | 64 | 124.5 | 299.5 | 2.41× | 1211.9 | 2918.3 | 2.41× |
| Q4_K | 128 | 125.1 | 301.2 | 2.41× | 1212.2 | 2943.8 | 2.43× |
| Q4_K | 512 | 123.4 | 293.9 | 2.38× | 1190.1 | 2808.7 | 2.36× |
| Q6_K | 1 | 33.6 | 65.9 | 1.96× | 291.0 | 416.2 | 1.43× |
| Q6_K | 8 | 74.8 | 269.9 | 3.61× | 725.1 | 2360.2 | 3.25× |
| Q6_K | 16 | 76.2 | 288.9 | 3.79× | 741.9 | 2643.6 | 3.56× |
| Q6_K | 64 | 77.0 | 299.2 | 3.89× | 756.3 | 2916.0 | 3.86× |
| Q6_K | 128 | 77.1 | 301.8 | 3.91× | 758.8 | 2944.1 | 3.88× |
| Q6_K | 512 | 77.0 | 293.7 | 3.81× | 750.8 | 2804.7 | 3.74× |
| Q5_0 | 1 | 21.0 | 69.4 | 3.30× | 202.0 | 473.6 | 2.34× |
| Q5_0 | 8 | 54.8 | 277.3 | 5.06× | 537.7 | 2374.8 | 4.42× |
| Q5_0 | 16 | 55.1 | 290.0 | 5.26× | 537.2 | 2643.2 | 4.92× |
| Q5_0 | 64 | 55.5 | 300.4 | 5.41× | 547.4 | 2882.7 | 5.27× |
| Q5_0 | 128 | 55.5 | 303.5 | 5.47× | 546.9 | 2896.3 | 5.30× |
| Q5_0 | 512 | 55.2 | 293.1 | 5.31× | 538.3 | 2875.5 | 5.34× |

### 8192 × 2048: 8 192 weights × 2 048 rows, 256 tiles streamed

**sdot tier**

| Format | Rows | exact 1T | form 1T | ratio | exact 10T | form 10T | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 1 | 45.4 | 31.7 | 0.70× | 231.4 | 135.6 | 0.59× |
| Q4_K | 8 | 72.8 | 125.0 | 1.72× | 675.2 | 855.7 | 1.27× |
| Q4_K | 16 | 74.7 | 148.7 | 1.99× | 715.7 | 1269.6 | 1.77× |
| Q4_K | 64 | 76.5 | 172.2 | 2.25× | 731.0 | 1651.4 | 2.26× |
| Q4_K | 128 | 76.5 | 176.7 | 2.31× | 728.1 | 1707.9 | 2.35× |
| Q4_K | 512 | 77.2 | 182.0 | 2.36× | 691.3 | 1525.6 | 2.21× |
| Q6_K | 1 | 31.9 | 32.4 | 1.02× | 133.6 | 144.3 | 1.08× |
| Q6_K | 8 | 49.4 | 126.9 | 2.57× | 463.0 | 875.9 | 1.89× |
| Q6_K | 16 | 50.6 | 148.7 | 2.94× | 491.3 | 1291.1 | 2.63× |
| Q6_K | 64 | 51.7 | 172.3 | 3.33× | 496.6 | 1658.6 | 3.34× |
| Q6_K | 128 | 51.8 | 176.3 | 3.40× | 495.3 | 1716.6 | 3.47× |
| Q6_K | 512 | 52.0 | 182.1 | 3.50× | 478.8 | 1532.4 | 3.20× |
| Q5_0 | 1 | 21.2 | 32.3 | 1.52× | 143.9 | 133.4 | 0.93× |
| Q5_0 | 8 | 33.6 | 127.2 | 3.79× | 317.6 | 928.8 | 2.92× |
| Q5_0 | 16 | 33.9 | 148.9 | 4.39× | 315.6 | 1316.9 | 4.17× |
| Q5_0 | 64 | 34.2 | 172.4 | 5.04× | 318.3 | 1660.2 | 5.22× |
| Q5_0 | 128 | 34.0 | 177.1 | 5.21× | 316.4 | 1720.9 | 5.44× |
| Q5_0 | 512 | 33.9 | 181.9 | 5.37× | 308.9 | 1526.8 | 4.94× |

**i8mm tier**

| Format | Rows | exact 1T | form 1T | ratio | exact 10T | form 10T | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K | 1 | 46.5 | 31.8 | 0.68× | 220.2 | 138.3 | 0.63× |
| Q4_K | 8 | 116.4 | 157.8 | 1.36× | 936.3 | 1007.2 | 1.08× |
| Q4_K | 16 | 119.4 | 205.6 | 1.72× | 1119.7 | 1649.2 | 1.47× |
| Q4_K | 64 | 123.5 | 269.0 | 2.18× | 1195.3 | 2514.8 | 2.10× |
| Q4_K | 128 | 123.6 | 284.0 | 2.30× | 1198.5 | 2723.8 | 2.27× |
| Q4_K | 512 | 124.9 | 292.4 | 2.34× | 1117.2 | 2225.1 | 1.99× |
| Q6_K | 1 | 30.5 | 32.3 | 1.06× | 140.2 | 149.8 | 1.07× |
| Q6_K | 8 | 70.9 | 157.6 | 2.22× | 613.9 | 1075.3 | 1.75× |
| Q6_K | 16 | 73.6 | 205.5 | 2.79× | 694.9 | 1731.2 | 2.49× |
| Q6_K | 64 | 76.4 | 269.8 | 3.53× | 746.5 | 2552.1 | 3.42× |
| Q6_K | 128 | 76.4 | 283.8 | 3.71× | 743.0 | 2744.3 | 3.69× |
| Q6_K | 512 | 76.9 | 292.1 | 3.80× | 717.1 | 2308.5 | 3.22× |
| Q5_0 | 1 | 21.3 | 32.0 | 1.50× | 143.9 | 159.5 | 1.11× |
| Q5_0 | 8 | 53.2 | 157.3 | 2.96× | 493.8 | 1111.8 | 2.25× |
| Q5_0 | 16 | 54.3 | 206.4 | 3.80× | 520.8 | 1806.9 | 3.47× |
| Q5_0 | 64 | 55.4 | 270.5 | 4.88× | 535.0 | 2546.4 | 4.76× |
| Q5_0 | 128 | 54.9 | 283.2 | 5.16× | 531.9 | 2701.7 | 5.08× |
| Q5_0 | 512 | 55.4 | 292.0 | 5.27× | 525.0 | 2445.5 | 4.66× |
## Where the form wins

- **Every width of 8 activation rows and above**, on both shapes and both tiers.
  At 4 096 × 512 the form is 2.07–2.40× the exact Q4_K kernel on `sdot` and
  2.06–2.43× on `smmla`; Q6_K is 3.16–3.54× and 3.25–3.88×; Q5_0 is 4.89–5.67×
  and 4.42–5.30×. At 8 192 × 2 048, streamed, the same cells are 1.27–2.35×
  (Q4_K), 1.89–3.47× (Q6_K) and 2.92–5.44× (Q5_0) on `sdot`, and 1.08–2.34×,
  1.75–3.69× and 2.25–5.08× on `smmla`. The smallest win in the table is Q4_K on
  `smmla` at 8 rows of the streamed shape, 1.08×.
- **Not at one row.** A one-row pass is bandwidth-bound, and the form's 1.03
  bytes per weight is 1.8× Q4_K's 0.5625: on the streamed shape the Q4_K form is
  0.68–0.70× at one thread and 0.63× at ten (`smmla`), and the Q5_0 form is
  0.93× on `sdot` at ten. Q6_K stays level (1.02–1.08×) and Q5_0 on `smmla` is
  1.11–1.50×, because their exact kernels are slower. On the resident
  4 096-weight shape, where the exact tile is in L2, the form is still ahead at
  one row for `sdot` Q4_K/Q6_K/Q5_0 (1.20×/1.91×/2.07×); at ten threads the
  wider bandwidth share makes Q4_K `sdot` 0.89× and `smmla` 1.01×.
- The resident form makes Q5_0's exact fold disappear: its exact `sdot` kernel is
  21–35 GMAC/s, the form runs at the same 176–182 GMAC/s as every other format.
  Q6_K's exact kernel is 31–77 and Q4_K's 45–125; the form is one kernel at
  176–182 (`sdot`) or 292–303 (`smmla`) GMAC/s per core.
- **Ten threads scale rather than saturate** on the L2-resident shape
  (9.5–9.8× the one-thread figure); on the streamed shape the form reaches
  2.23–2.45 TMAC/s on `smmla` against 1T's 292–303 GMAC/s, 7.6–8.1×.

## Memory on Llama 3.2 1B

`cargo run --release -p peregrine-bench --example kq8model -- errors
~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` reports 1 235 746 816 weights in the
Q4_K/Q6_K/Q5_0 tensors and a **resident form of 1 274 363 904 bytes (1.187 GiB,
1.0312 bytes per weight)**. The whole file is 799 592 448 bytes (0.745 GiB), so
the form costs +0.44 GiB on the 1B — the same number S5.4a measured on x86, as
it must be: the form's geometry does not depend on the tier. That 1.187 GiB is the form for all
three formats, which is what S5.4a's table measures. The option as built by S5.4b carries Q4_K
and Q6_K only, and that form is **957.0 MiB** on the same file (S5.4b's measurement; the
geometry is tier-independent, so the ARM figure is the same).

## What surprised

The 8-bit kernel's ceiling is much higher on this core than the AVX-512 desktop
slice suggested: 176–182 GMAC/s for `sdot` and 292–303 for `smmla` per Cortex-X925
at 512 rows, where S5.4a measured 91.9 on Zen 5. The exact kernels are also
faster here (Q4_K 77 on `sdot`, 125 on `smmla`), so the *relative* win is smaller
than the desktop's (2.3× against 1.4× for Q4_K at 512 rows), and the exact Q5_0
kernel is the outlier that makes the form look 5× better. The build is vectorised
for load time, not because it is in the token path; S5.4a's per-pass conversion
row is why the form stays resident.
