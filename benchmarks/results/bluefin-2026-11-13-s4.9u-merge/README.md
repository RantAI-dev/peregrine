# S4.9u on the merged build: the batch cells of nomic-bert and jina-bert-v2 (2026-11-13)

`docs/benchmark.md`'s two new family rows are the S4.9u run on `773ebf8` plus S4.9u's three
same-bits commits. Since then `#251` (the `nomic-bert-moe` family, S4.8b) landed on `main`
and changed one thing these rows depend on: a **float-only** bidirectional encoder whose
file is over 256 MiB now carries **128 rows** per pass instead of 32
(`peregrine_core::options::pass_rows_default_for`). nomic-embed-text v1.5 F16 (274 MB) and
jina-embeddings-v2-base-en F16 (274 MB) are both just over that line, so their **batch**
cells — the ones that put several texts in one pass — were re-measured on the merged build.
Single-text cells (`b1`) do not involve the rule (a text is never split) and were not
re-run, nor were the Q8_0 files (146 MB, not float-only: the rule keeps 32 rows there).

The 128-token cells do not move either, and that is the rule's own arithmetic: a pass of
128 rows holds **one** 128-token text, so 32 and 128 rows per pass buy the same thing. The
16-token cells are where it pays — a 32-row pass holds two 16-token texts, a 128-row pass
holds eight — which is the 16 × b8/b32 cells of both models.

**What moved.** nomic F16's two 16-token batch cells flip from behind to ahead
(0.936 → **1.019×**, 0.979 → **1.088×**), its 128-token b8 cell becomes level
(0.946 → **1.002×**), and the 512-token and mixed cells are unchanged inside their spread.
jina F16's 16-token cells narrow (0.889 → **0.938×**, 0.891 → **0.964×**) but stay behind;
its 128-, 512-token and mixed cells are unchanged. The count on the table's rule (Peregrine
not slower): **24 of the 36 cells this pair adds, 79 of 104 with the existing table** (21 and
76 before this re-read), **twelve cells behind** (fifteen before).

## Machine, build and host state

| | |
|---|---|
| machine | bluefin: AMD Ryzen 3 PRO 4350G (Renoir, Zen 2), 4 cores / 8 threads, 128 KiB L1d + 512 KiB L2 per core, 4 MiB L3, 13 GB, Fedora 44, kernel 7.2.7-200.fc44, governor `powersave` |
| flags | `fma avx avx2` (tier `x86_64-avx2`), 4 threads, cache off (`PEREGRINE_EMBED_CACHE_MB=0`) |
| build | `2215445` — the merge of `origin/main` (`be6eea8`…`3279a47`) into `s4.9u-new-families-standings`; `cargo build --release -p peregrine` |
| before | the numbers in `benchmarks/results/bluefin-2026-11-12-s4.9u/README.md`, taken on `773ebf8` + S4.9u's three commits |
| competitors | ONNX Runtime 1.31.1, fastembed 0.9.1, OpenVINO 2026.4.1 — the fp32 rows the S4.9u run used for these cells |
| host | **not quiet**: `/proc/loadavg` 13–40 (1-minute) in every window, but **CPU idle 52.7 % and 53.1 % of two of the three rounds — exactly the measurement's own four threads on eight logical CPUs — with steal 0 and iowait 0.3–0.5 %** (`/proc/stat` deltas over the whole round, `s49u-merge-cpu-r{2,3}.txt`). The load average here is host-wide inside this session's PID namespace (the host keeps ~40 D-state tasks on a 98 %-full disk that are not ours and use no CPU); the harness's load gate was therefore left open (`--max-load 1000`) and the CPU share recorded instead. Swap 7.4 GB of 8.0 GB in use, the same condition S4.9l–S4.9u recorded. |

## Method

- The S4.9u batch cells only: 16/128/512 tokens × batch 8 and 32, plus the mixed-length
  batch of 32 texts (8–512 tokens, seed `20261111`), for nomic F16 and jina F16 against the
  best equal-accuracy engine of that run. Single-text cells and the Q8_0 files: not re-run.
- The same `benchmarks/embed.py` settings as S4.9u (`--budget 0.5 --max-requests 20
  --max-big-requests 2`, one client, `--cells` selecting the seven cells), fresh texts inside
  a batch and per timed request, three interleaved rounds (all rows round 1, then 2, then 3),
  one engine and one model process at a time, medians of the three rounds.
- ONNX Runtime's nomic row is measured one cell per invocation: on this host its server
  drops the connection between cells of one run (the same rows did run whole in S4.9u).
  Each cell still comes from its own server start and its own request sequence, like every
  other row; `parts/nomic-ort-r{1,2,3}/c*/` keeps them, and the merged row is
  `parts/nomic-ort-r{1,2,3}/nomic-onnxruntime-fp32-r*.json`.
- Accuracy is not re-measured: no engine's file or arithmetic changed, and the same fp32
  engines stay in the equal-accuracy class (nomic/jina F16 cosine 1.000000 in S4.9u).

## Peregrine F16 against the best equal-accuracy engine

Medians of the three interleaved rounds, texts/s; the ratio is Peregrine ÷ the best other
engine on the same cell. "before" is the S4.9u figure for the same cell.

**nomic-embed-text v1.5 F16**

| cell | before | now | Peregrine | best other | its rounds |
|---|---:|---:|---:|---|---|
| L16 b8 | 0.936× | **1.019×** | 85.37 | ONNX Runtime fp32 83.75 | 83.4–86.1 |
| L16 b32 | 0.979× | **1.088×** | 86.77 | ONNX Runtime fp32 79.76 | 86.3–86.9 |
| L128 b8 | 0.946× | **1.002×** (level) | 10.95 | fastembed fp32 10.93 | 10.92–11.04 |
| L128 b32 | 0.947× | 0.929× | 10.92 | fastembed fp32 11.75 | 10.75–10.91 |
| L512 b8 | 1.078× | **1.090×** | 2.515 | OpenVINO fp32 2.307 | 2.50–2.54 |
| L512 b32 | 1.113× | **1.102×** | 2.498 | fastembed fp32 2.267 | 2.50–2.54 |
| mixed b32 | 1.708× | **1.717×** | 12.16 | fastembed fp32 7.08 | 11.58–12.23 |

**jina-embeddings-v2-base-en F16**

| cell | before | now | Peregrine | best other | its rounds |
|---|---:|---:|---:|---|---|
| L16 b8 | 0.889× | 0.938× | 85.52 | ONNX Runtime fp32 91.17 | 85.1–85.7 |
| L16 b32 | 0.891× | 0.964× | 88.21 | ONNX Runtime fp32 91.48 | 88.1–88.4 |
| L128 b8 | 0.916× | 0.917× | 10.99 | fastembed fp32 11.99 | 10.97–11.10 |
| L128 b32 | 0.920× | 0.910× | 11.05 | fastembed fp32 12.14 | 10.83–11.06 |
| L512 b8 | 1.014× | **1.002×** | 2.574 | OpenVINO fp32 2.569 | 2.56–2.59 |
| L512 b32 | 1.031× | **1.042×** | 2.579 | OpenVINO fp32 2.474 | 2.58–2.58 |
| mixed b32 | 1.666× | **1.692×** | 12.42 | fastembed fp32 7.34 | 12.36–12.45 |

The mixed cell's Peregrine rounds spread 5.3 % (nomic) and 0.8 % (jina); every other cell is
inside 0.3–2.1 %. jina's L512 b8 is level (1.002×, its own rounds 2.56–2.59 against
OpenVINO's 2.569), as is nomic's L128 b8.

## The full after table, texts/s (median of three rounds)

| engine / file | L16 b8 | L16 b32 | L128 b8 | L128 b32 | L512 b8 | L512 b32 | mixed b32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **nomic** | | | | | | | |
| peregrine-f16 | 85.37 | 86.77 | 10.95 | 10.92 | 2.515 | 2.498 | 12.16 |
| onnxruntime-fp32 | 83.75 | 79.76 | 9.58 | 9.35 | 2.126 | 2.121 | 2.06 |
| fastembed-fp32 | 60.29 | 62.80 | 10.93 | 11.75 | 2.247 | 2.267 | 7.08 |
| openvino-fp32 | 79.94 | 78.69 | 8.47 | 9.39 | 2.307 | 2.223 | 2.29 |
| **jina** | | | | | | | |
| peregrine-f16 | 85.52 | 88.21 | 10.99 | 11.05 | 2.574 | 2.579 | 12.42 |
| onnxruntime-fp32 | 91.17 | 91.48 | 10.52 | 9.82 | 2.181 | 2.129 | 2.14 |
| fastembed-fp32 | 64.23 | 68.23 | 11.99 | 12.14 | 2.305 | 2.307 | 7.34 |
| openvino-fp32 | 86.76 | 83.85 | 9.88 | 10.47 | 2.569 | 2.474 | 2.55 |

(Nomic's OpenVINO row is `sentence-transformers+openvino-fp32` in the raw JSONs, the same
engine the S4.9u table lists as OpenVINO.)

## The cells below 1.0 after the re-read, all of them named

| model / file | cells | ratio |
|---|---|---|
| nomic F16 | L128 b1 (not re-run), L128 b32 | 0.929, 0.929 |
| jina F16 | L16 b8, L16 b32 | 0.938, 0.964 |
| jina F16 | L128 b1 (not re-run), b8, b32 | 0.911, 0.917, 0.910 |
| jina Q8_0 (not re-run) | L16 b8, L16 b32 | 0.945, 0.974 |
| jina Q8_0 (not re-run) | L128 b1, b8, b32 | 0.916, 0.903, 0.918 |

Twelve cells, and every one of them is a 128-token cell or a 16-token batch cell: the
128-token cells need the matrix product's small-row rate (S4.9v's account), and the 16-token
batch cells are the same block seen through more texts.

**The count.** With the table's rule (Peregrine not slower): nomic F16 10 of 12, jina F16 7
of 12, jina Q8_0 7 of 12 — **24 of the 36 cells these two models add** (21 before), i.e.
**79 of 104** on `docs/benchmark.md` (76 before). If the two level cells (nomic L128 b8
1.002×, jina L512 b8 1.002×) are not counted as wins, it is 22 of 36 and 77 of 104; both
cells are inside the run-to-run spread.

## Reproducing

```sh
export PEREGRINE_EMBED_CACHE_MB=0
python benchmarks/embed.py --select nomic:peregrine:f16 --round 1 --threads 4 \
  --texts-dir /var/tmp/embed-texts-s49u --output-dir <dir> \
  --peregrine-binary ./target/release/peregrine --lengths 16,128,512 \
  --batches 8,32 --clients 1 --budget 0.5 --max-requests 20 --max-big-requests 2 \
  --mixed-cell --mixed-batch 32 \
  --cells L16_b8_c1,L16_b32_c1,L128_b8_c1,L128_b32_c1,L512_b8_c1,L512_b32_c1,mixed_b32_c1
# …the same for the other 7 rows (fastembed, onnxruntime, openvino per model), rounds 2 and 3
# the CPU share over a round: awk '/^cpu /' /proc/stat before and after the loop
```

The raw JSONs are `raw/` (seven rows × three rounds) and `parts/nomic-ort-r*/` (the ONNX
Runtime row, one cell per invocation); `s49u-merge-cpu-r{2,3}.txt` holds the `/proc/stat`
lines the idle share is computed from.
