# The Qwen3-Embedding long-text "slowdown": it does not reproduce

S4.9q's report had Qwen3-Embedding 0.6B Q8_0 at one 2 000-token text falling from
0.213 texts/s in the 2026-11-12 standings run (`3030172`) to 0.177 when `3030172` was
rebuilt, and to 0.145 on main before the `i8mm` kernel. This run rebuilt both commits
side by side and measured them interleaved, on the same host, with the same harness and
flags. **Main is not slower than `3030172` in any cell, on a quiet host or on a loaded
one.** The whole host moves by a factor of two between those two states, and every
number in the earlier trio is a loaded-host number for the *same* old commit.

No code changes, and no bisect: the bisect criterion (a step more than 8 % below
`3030172`'s median) is never met, so there is no bad commit to find.

## Host and method

- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores (10 × Cortex-X925 +
  10 × Cortex-A725), 121 GB unified memory, governor `performance`, kernel tier
  `aarch64-neon-i8mm`.
- Arms: `3030172` (the standings commit, checked out in its own worktree with its own
  target directory) and `origin/main` at `b53ec7f`. Both built with `cargo build --release
  -p peregrine`, rustc 1.99.1.
- Harness: `benchmarks/embed.py`, `/v1/embeddings`, one server per round, 10 threads
  pinned to the Cortex-X925 cluster with `taskset -c 5-9,15-19`, five interleaved rounds
  per arm per cell (old, main, old, main, ...), medians. Embedding cache off
  (`PEREGRINE_EMBED_CACHE_MB=0`). Texts are the ones `benchmarks/embed_texts.py` wrote
  from the model's own tokenizer (`/var/tmp/embed-texts/qwen3-embed-0.6b.json`), a
  distinct text per request, so no cell repeats a text and the exact cache cannot
  contribute.
- Co-tenant load: stated per table below. The host carries two `VLLM::EngineCore`
  processes and other tenants; between 12:20 and 12:28 local the 1-minute load average
  was 1.6 and between 12:31 and 12:50 it was 8.2–9.1. Both arms were measured inside the
  same window, alternating.

## Reproduction, quiet host (load average 1.6, 36 GB available)

Cells measured: the single 2 000-token text, the 512- and 16-token single texts, and
the batch cell (32 texts of 128 tokens).

| Cell | `3030172` texts/s | `main` texts/s | main ÷ old | `3030172` spread | `main` spread |
|---|---|---|---|---|---|
| L16 b1 | 60.834 | 62.363 | **1.025×** | 60.281–61.138 | 59.162–64.854 |
| L512 b1 | 1.767 | 1.849 | **1.046×** | 1.751–1.783 | 1.828–1.867 |
| L2000 b1 | 0.360 | 0.380 | **1.056×** | 0.359–0.363 | 0.299–0.383 |
| L128 b32 | 4.275 | 4.488 | **1.050×** | 4.249–4.307 | 4.443–4.542 |

The 0.299 in main's L2000 spread is one round out of five; the other four are
0.378–0.383, and its L16 spread has the same shape (59.162 and 64.854 against
62.198–63.836). Those two rounds are the co-tenant arriving mid-round; they are in the
spread rather than smoothed away.

## Reproduction, loaded host (load average 8.2–9.1, 36 GB available, two vLLM tenants)

| Cell | `3030172` texts/s | `main` texts/s | main ÷ old | `3030172` spread | `main` spread |
|---|---|---|---|---|---|
| L16 b1 | 30.897 | 32.710 | **1.059×** | 29.551–31.528 | 32.198–33.144 |
| L512 b1 | 0.880 | 0.951 | **1.081×** | 0.876–0.889 | 0.933–0.953 |
| L2000 b1 | 0.179 | 0.191 | **1.067×** | 0.178–0.181 | 0.191–0.192 |
| L128 b32 | 3.983 | 4.224 | **1.061×** | 3.938–3.993 | 4.183–4.238 |

The loaded table keeps the same ordering. The single-text cells run at 50–52 % of their
quiet throughput, while the 32-text batch holds 93 % of its own (3 983 against 4 275) — a
batch amortises the weight passes, so it suffers less from a co-tenant taking cores and
bandwidth. `3030172` at 0.179 is the "0.177 when I rebuilt it" number to within 1 %, and
main at 0.191 is above it.

## The reported trio

| Number | What it was | What it is |
|---|---|---|
| 0.213 | `3030172`, standings run 2026-11-12 00:00–02:30 | that run's own record shows loadaverage 6.8–8.5 and 35 GB available for a 10-thread cell; a loaded-host number for the same commit |
| 0.177 | `3030172` rebuilt | today `3030172` measures 0.179 under the same class of load, and 0.360 quiet |
| 0.145 | main before the `i8mm` kernel | today main measures 0.191 loaded and 0.380 quiet; 0.145 is where the *old* commit also lands when the co-tenant spikes (one main round today hit 0.299, one chunk-flip round 0.15) |

The 0.213 → 0.177 step happened with no source change at all, which is the clearest
evidence that the trio is host state, not commit history: the same commit, rebuilt
twice, produced both ends of it.

## Bisect log

Not opened. A bisect step would be a commit more than 8 % below `3030172`'s median;
main is 2.5–8.1 % *above* `3030172` in every one of the eight cells above, so there is
no first bad commit, and no second one. What was done instead is the change review
below, which covers the causal prefill path exhaustively.

## What changed on the causal prefill path, 3030172 → main

Qwen3-Embedding 0.6B is a causal embedder: it runs the causal prefill path, not the
bidirectional encoder one. Between the two commits:

- **Kernel selection for Q8_0 is identical.** `kernels_for(Q8_0)` on aarch64 is
  `[Scalar, NeonDotprod]` in both trees — the same `sdot` tile kernel, no route change.
  (S4.9o's float-panel route is scoped to the bidirectional encoder path; it does not
  reach this model.) The `i8mm` tile body is in #235, which is not on main yet, so both
  arms here run the `sdot` body.
- **The pass budget for embedding work is identical.** `pass_rows_default_for`'s causal
  branch (`rows.clamp(16, 64)`) is byte-for-byte the same in both trees; only the
  bidirectional branch gained the S4.9m/S4.9o floors, and that branch is not taken here.
- **Threading for small work is unchanged** (PR #188 predates `3030172`).
- **The one change that does touch this path is the wide prefill pass (S5.1, #199)**:
  when the pass carries no decoding row, main reads the prompt at `wide_chunk` = 64
  instead of `prefill_chunk` = 16 (`serve.rs`, the `dec.is_empty()` arm). That makes a
  long single-text pass wider, not narrower, and it is the only difference.

## Confirming the one difference by flipping it back

`PEREGRINE_PREFILL_CHUNK=16` on main forces the old chunk (an explicit chunk is also the
wide one). Five interleaved rounds on the 2 000-token cell, loaded host:

| main, 2 000-token cell | median texts/s | spread |
|---|---|---|
| default (wide chunk 64) | 0.19 | 0.15–0.19 |
| `PEREGRINE_PREFILL_CHUNK=16` | 0.19 | 0.19–0.19 |

Neutral. The wide pass is not the cause of anything measured here, in either direction;
on this model a 64-row pass and a 16-row pass cost the same, so the S5.1 change neither
explains the reported fall nor offers a fix.

## Stage profile (supplementary)

`crates/peregrine-bench/examples/cellprof` (the instrumented `Engine`, one slot, one
2 000-token text, 10 threads pinned, medians of three rounds, loaded host):

| stage (ms) | `3030172` | `main` |
|---|---|---|
| matmul | 3235 | 3242 |
| attention | 1640 | 1777 |
| head | 0 | 0 |
| other | 463 | 448 |
| total | 5338 | 5464 |

Attention reads 8.3 % higher and total 2.4 % higher on main in *this* harness, while the
same cell over the server path reads 5.6 % *lower* on main. Two things about that:
`cellprof` drives a one-slot engine with the step counters on, and the server drives a
40-slot engine with `--no-profile`; the two harnesses disagree by about 9 % on this
cell, and the instrumented one is the one that is not the graded path. The profile's
useful part is the shape: matmul and the non-attention remainder are flat to 1 %, so no
kernel regressed; attention is where any drift would be, and it is inside the harness
band.

## What this means for the S4.9q report

- There is no commit to bisect and no fix to ship. The 0.213/0.177/0.145 trio cannot be
  reproduced as a commit effect: the same commit spans 0.179 (loaded) to 0.360 (quiet)
  today.
- The absolute numbers in the 2026-11-12 embeddings standings for this cell were taken
  under a co-tenant that halved the host. Any later comparison against that table has to
  be made inside one window, with both arms, not against the table.
- One related caveat found while setting this up: the standings table's *batch* cells
  for Peregrine are not cache-free. The batch cell runs immediately after the batch-1
  cell on the same process, and the 64 MiB exact cache serves the texts batch 1 just
  computed (20 of 32 at L128), which is why that table has L128 b32 at 10.52 texts/s
  where a cache-off run of either commit gives 4.0–4.5. Checked: the same sequence with
  the cache left on gives 10.74 texts/s (2 978.9 ms for the 32-text request) against
  3.98–4.27 with `PEREGRINE_EMBED_CACHE_MB=0`. This does not affect the single-text
  cells this report is about, and every table above is cache-off.

## Exact commands

`~` stands for the home directory. Both arms in separate worktrees and target
directories, the harness pinned to the Cortex-X925 cluster:

```sh
git worktree add --detach /var/tmp/pg/work/qwen3-old 3030172
git worktree add -b fix-qwen3-embed-long-text ~/wt-qwen3fix origin/main
cd /var/tmp/pg/work/qwen3-old && CARGO_TARGET_DIR=/var/tmp/pg/target-old \
    cargo build --release -p peregrine
cd ~/wt-qwen3fix && CARGO_TARGET_DIR=/var/tmp/pg/target-main \
    cargo build --release -p peregrine
```

One round of one arm (alternate the two arms and repeat for five rounds):

```sh
PEREGRINE_EMBED_CACHE_MB=0 taskset -c 5-9,15-19 \
python3 benchmarks/embed.py --select qwen3-embed-0.6b:peregrine:q8_0 --round 1 \
    --lengths 16,512,2000 --batches 1 --clients 1 --threads 10 \
    --budget 30 --max-requests 4 --max-big-requests 1 --max-cell-seconds 300 \
    --max-load 20 --ctx-tokens 2048 \
    --peregrine-worktree /var/tmp/pg/work/qwen3-old \
    --peregrine-binary /var/tmp/pg/target-old/release/peregrine \
    --texts-dir /var/tmp/embed-texts --output-dir /var/tmp/pg/q3-a/old-pre1
```

The batch cell is the same command with `--lengths 128 --batches 32`; the chunk flip adds
`PEREGRINE_PREFILL_CHUNK=16`; the stage profile is

```sh
cd /var/tmp/pg/work/qwen3-old && CARGO_TARGET_DIR=/var/tmp/pg/target-old \
  cargo build --release -p peregrine-bench --example cellprof \
  --features peregrine-models/arch-qwen
taskset -c 5-9,15-19 /var/tmp/pg/target-old/release/examples/cellprof \
  ~/models/embed/qwen3-embed-0.6b/qwen3-embed-0.6b-q8_0.gguf \
  /var/tmp/embed-texts/qwen3-embed-0.6b.json 2000 1 10
```

Model file: `~/models/embed/qwen3-embed-0.6b/qwen3-embed-0.6b-q8_0.gguf`, SHA-256
`06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`.
