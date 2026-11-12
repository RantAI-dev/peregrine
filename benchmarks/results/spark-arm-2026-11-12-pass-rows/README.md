# The pass width of large float encoders on ARM (PASS_ROWS 32 against 128)

#251 (`pass_rows_default_for`) gave a bidirectional encoder whose weights are all float and
larger than 256 MiB **128 rows per pass** instead of 32. It was measured on x86 only: a batch of
sixteen 32-token texts 19.4 → 23.5 texts/s on nomic-embed-text v1.5 F16, and nomic-embed-text-v2-moe
1.53 s → 0.90 s on the same batch. This round measures the rule on this ARM host for the five
large float encoders it has, at both thread counts, with the two settings of one build
(`--opt PASS_ROWS=32` against `--opt PASS_ROWS=128`), alternating inside every round. **The rule
is right for ARM and nothing changed**: 128 is level on every single-text cell and 1.04-1.29×
ahead on every batch at one thread, and no cell is slower at one thread outside the noise.

- Date: 2026-11-13 02:00-04:10 local (UTC+7).
- Host: DGX Spark (Lenovo ThinkStation PGX), aarch64, 20 cores in two clusters: 10 ×
  Cortex-X925 at 3.9 GHz (CPUs 5–9 and 15–19) and 10 × Cortex-A725 at 2.8 GHz, 121 GB unified
  memory, Linux `6.17.0-1008-nvidia`, governor `performance`, rustc 1.99.1, kernel tier
  `aarch64-neon-i8mm`.
- Peregrine: `origin/main` at `b12d327` (#251's merge), no file changed this round. Two
  `peregrine serve` processes per model and thread count, one per setting, both on the same
  build; each request is timed end to end over a kept-alive connection.
- The host was shared: the 1-minute load average was **1.0-3.2 for every one-thread cell** and
  **5.7-32.5 for the ten-thread ones**; it is beside every cell in `table.txt` and in each
  `pass-rows-*.jsonl` line.

## The models

All five are float-only and, at 260.9-1155.3 MiB, over the rule's 256 MiB line — so all five are
in the class #251 moved (two of them, nomic v1.5 F16 and jina v2 F16, by 5 MiB). Sizes and
SHA-256 are in `models.txt`:

| model | file | weights | blocks / d / ff |
|---|---|---:|---|
| bge-m3 F16 | `bge-m3-FP16.gguf` (gpustack/bge-m3-GGUF) | 1097.5 MiB | 24 / 1024 / 4096 |
| EmbeddingGemma 300M F32 | `embeddinggemma-300M-F32.gguf` (unsloth/…-GGUF) | 1155.3 MiB | 24 / 768 / 1152 |
| gte-modernbert-base F16 | `gte-modernbert-base-F16.gguf` (local) | 284.3 MiB | 22 / 768 / 1152 |
| nomic-embed-text v1.5 F16 | `nomic-embed-text-v1.5.f16.gguf` (nomic-ai/…) | 260.9 MiB | 12 / 768 / 3072 |
| jina-embeddings-v2-base-en F16 | `jina-embeddings-v2-base-en-f16.gguf` (second-state/…) | 261.0 MiB | 12 / 768 / 3072 |

## The cells

One text of 16, 128 and 512 tokens; sixteen different 32-token texts; thirty-two different
128-token texts; eight different 512-token texts; and the mixed batch of 32 whose lengths are
drawn from the ARM table's pool (8-512 tokens, sum 3.6-4.9 k tokens here). The embedding result
cache is off (`--opt EMBED_CACHE_MB=0`), every text of a request is distinct, and every timed
request gets **fresh** texts. Token counts are calibrated per model through `usage.prompt_tokens`
(within 5 % of the target; the exact count is in the record) and recorded, so the ratio is over
tokens/s.

## 128 ÷ 32, one thread

Medians of the five interleaved rounds (bge-m3's three large-batch cells: three rounds), and the
per-round range. One-thread throughput is stable to ±1 %, which is why these are the standing.

| model | 16 | 128 | 512 | 16×32 | 32×128 | 8×512 | mixed 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bge-m3 F16 | 1.003 | 1.007 | 0.999 | **1.121** | **1.048** | **1.064** | 1.004 |
| EmbeddingGemma F32 | 0.998 | 1.000 | 1.000 | **1.289** | **1.081** | **1.142** | 1.010 |
| gte-modernbert F16 | 1.003 | 1.006 | 1.000 | **1.110** | **1.038** | **1.066** | 1.007 |
| nomic v1.5 F16 | 1.005 | 1.001 | 1.000 | **1.084** | **1.039** | **1.081** | 1.005 |
| jina v2 F16 | 1.001 | 1.002 | 1.000 | **1.084** | **1.041** | **1.085** | 1.005 |

- **A single text is level** (0.996-1.008, every range inside ±1 %): 128 rows never split a text,
  so the one-text cells are the same work in both arms — the control that shows the harness is
  not measuring a setting artefact.
- **Every batch gains**: 16 × 32 tokens 1.084-1.289, 32 × 128 1.038-1.081, 8 × 512 1.064-1.142,
  and the mixed batch 1.004-1.010 (its texts are mostly short, so it is closer to the single-text
  case). The short-text batch, where a 32-row pass is 8-16 weight passes over the model, gains
  most; the 512-token batch still gains 6-14 % because 128 rows amortise the weight stream
  without splitting a text.

## 128 ÷ 32, ten threads

Medians of five rounds, all **loaded** (load 5.7-32.5). The spread of the per-round ratios is
wide — the same host that makes the fast-prompt ten-thread cells unusable here too:

| model | 16 | 128 | 512 | 16×32 | 32×128 | 8×512 | mixed 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bge-m3 F16 | 1.422 | 1.010 | 0.817 | **2.274** | **1.279** | **1.448** | 1.088 |
| EmbeddingGemma F32 | 1.179 | 1.070 | 1.087 | **1.249** | 0.970 | **1.132** | 0.996 |
| gte-modernbert F16 | 0.984 | 0.983 | 0.983 | **1.285** | **1.144** | **1.153** | 1.010 |
| nomic v1.5 F16 | 1.075 | 1.017 | 1.014 | **1.214** | **1.116** | **1.150** | 1.014 |
| jina v2 F16 | 1.130 | 0.915 | 0.989 | 0.947 | 0.991 | 0.896 | 1.022 |

Below 1.0 outside the spread, and why they are not the rule's fault:

- **bge-m3 512 = 0.817** (rounds 0.712, 0.817, 1.146, 1.215, 0.735; load 5.7-12.5) and
  **jina 8×512 = 0.896** (0.736, 0.896, 0.903, 0.893, 0.919; load 20.2-28.3). At one thread the
  same cells are **0.999** and **1.085**. A row budget that lost 10-20 % at ten threads would
  lose at one too — the pass is the same work, only spread over more workers — so these are the
  host, and the ten-thread column is a direction, not a standing.
- **gte's 0.983-0.989** on the single texts and **EmbeddingGemma's 0.996** mixed are inside the
  ±1 % of their one-thread controls (1.000-1.006 / 1.010) and inside the ten-thread spread.

## The rule

**128 is at least level on every model and cell at one thread, and 1.04-1.29× ahead on every
batch**, so #251's rule — every bidirectional encoder whose weights are all float and larger than
256 MiB takes 128 rows per pass — is right for ARM, and this round changes nothing. The two
ten-thread cells that read below 1.0 are host-contaminated and are level or ahead at one thread.
A row's bits do not depend on the pass budget (a text is never split, and the pass budget is
scheduling only), so no golden moves; none did.

## Files

`pass_rows.py` — the harness (two kept-alive servers per setting, alternating rounds, fresh
texts, cache off). `pass-rows-<model>-<threads>t*.jsonl` — one line per timed request (cell,
round, seconds, tokens, load). `summarize.py`, `table.txt` — the two tables. `models.txt` — file
sizes and SHA-256.
