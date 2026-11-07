# Testing

```sh
cargo xtest       # = cargo test --workspace --release --locked
cargo xclippy     # = clippy on every target, warnings are errors
```

Tests run in release mode: kernels are tested as they ship, and debug-mode SIMD tests are slow.

## Three kinds of test

1. **Kernel tests** (unit tests in the kernel modules, `tests/simd.rs`, `tests/tern.rs`): each fast kernel
   against its scalar reference. They need no model files and always run.
2. **Synthetic model tests** (`tests/synthetic.rs`): tiny random GGUFs written at test time by
   `tests/support/` — llama and qwen3 in F32, F16, Q8_0, Q5_0, Q4_0, Q4_K and Q6_K, qwen2 (attention biases,
   NeoX rotary pairing) in F16 and Q8_0, llama with a Llama 3.x `rope_freqs.weight` table or linear rope
   scaling, and the qwen35 hybrid (Gated DeltaNet, gated attention, Hadamard rotation) in PQ2_0 and
   PTQ1_0, plus that hybrid with one Q8_0 MTP (NextN) block whose speculation must be lossless and a
   variant whose drafts are actually accepted. The same hybrid is written there without the rotation —
   a plain Qwen 3.5 file, in F16 and Q8_0 — and llama.cpp v0.9.0 loads it and gives the same greedy
   tokens. Each must load, give finite logits, be bit-identical across
   runs and across 1, 2 and 4 threads, decode greedily the same way twice, and match its golden for the
   current tier (`tests/golden/synthetic/<tier>/`). A `qwen3moe` file (8 experts, 2 used, F16 and Q8_0
   experts) is written there too: the mixture-of-experts family runs it — it loads, gives finite logits,
   is bit-identical across runs and across 1, 2 and 4 threads, matches its golden, and gives the same
   tokens alone and batched through `Engine` — and llama.cpp v0.9.0 loads and runs the same file. A
   `gpt-oss` file (4 experts, 2 used, F16 and MXFP4 experts, attention sinks, alternating
   sliding-window and full attention, YaRN rope scaling) is written there too, and llama.cpp v0.9.0
   gives the same greedy tokens for both. The
   same tests drive small models through `Engine`:
   prompts served alone must produce the same tokens when admitted together (a mixed prefill/decode step,
   both `Caches::Mapped` and, with `PEREGRINE_MIX=0`, `Caches::PerRow`), a shared opening must restore from
   the prefix store, and a LittleBit sidecar built in-test must give finite, deterministic logits. They
   need no model files, so they run in CI on every tier; the `x86_64-avx2` and scalar (`x86_64`) goldens
   were recorded on an Intel host and reproduced bit for bit on an AMD host.
   CI's pinned tiers (scalar, AVX2 and the AVX-512 tier under SDE) run with `PEREGRINE_GOLDEN=require`,
   so a new model or format must commit its goldens for all three; `PEREGRINE_GOLDEN=record` writes the
   missing ones (the AVX-512 ones on an AVX-512 machine).
   `tests/runtime_parity.rs` pins what the session and a one-slot engine must agree on, since ADR 0007
   made the engine the only runtime: the same prompt through `Session` and through a one-slot `Engine`
   (greedy: the same tokens; sampled: the session counts conversation turns into the sampling key and the
   engine does not, and the two per-slot settings — the turn and the KV rotation — are what make them
   agree), the taps a DSpark drafter needs (which must not change a logit bit), a second user arriving
   while the first one speculates, and the prefix store, which is off while a drafter is attached. The
   drafter it uses is written in the test (`tests/support/dspark.rs`: a `config.json` and a safetensors
   file), so it needs no model files either.
3. **Real model tests** (`tests/engine.rs`, `batch.rs`, `bonsai.rs`, `oracle.rs`, `dspark.rs`, ...): real
   GGUF files, checking tokens against reference outputs. They skip, with a message, when the file is
   missing. They are the release gate.

## Where tests look for model files

| File | Location |
|---|---|
| SmolLM2-360M-Instruct Q4_K_M, Qwen3-4B Q4_K_M | `$PEREGRINE_MODEL`, or `~/models/` |
| Qwen3-1.7B Q4_K_M | `models/` at the repository root |
| Qwen2.5-0.5B-Instruct Q4_K_M, Llama-3.2-1B-Instruct Q4_K_M | `~/models/` |
| Llama-2-7b-chat-hf Q4_K_M, Mistral-7B-Instruct-v0.2 Q4_K_M | `~/models/` |
| Gemma 3 1B Instruct Q4_K_M (`google_gemma-3-1b-it-Q4_K_M.gguf`) | `~/models/` |
| Qwen3-30B-A3B Q2_K (`Qwen3-30B-A3B-Q2_K.gguf`) | `~/models/` |
| Qwen3.5-2B Q4_K_M (`Qwen3.5-2B-Q4_K_M.gguf`) | `~/models/` |
| Phi-4 14B Q4_K_M (`phi-4-Q4_K_M.gguf`) | `~/models/` |
| Bonsai 2 9B PQ2_0 | `$PEREGRINE_BONSAI9`, or `models/bonsai2-9b/` |
| Bonsai 2 4B PQ2_0 with MTP | `$PEREGRINE_BONSAI4_MTP`, or `models/bonsai2-4b/` |
| TheBloke's Mistral 7B Instruct v0.2 and Llama 2 7B Chat Q4_K_M | `~/models/` (see below) |

`models/` is git-ignored. These locations will be unified under one variable as part of S0.1.

The SmolLM2 A1 oracle fixture is cut from one exact file: mradermacher's
`SmolLM2-360M-Instruct.Q4_K_M.gguf` (SHA-256 `ade7aee1e5bd7c3b8f1fedafc61a1a0be31e8afa64e006607cd202323e2ccce0`),
and `tests/oracle.rs` refuses any other upload.

### The two tokenizer oracle files

`tests/tokenize.rs` checks the tokenizer of two more models against token ids taken from llama.cpp's
`llama-tokenize`. It needs both files in `~/models/`, under exactly these names:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | <https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf> | 397808192 | `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` |
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | <https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf> | 807694464 | `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83` |

Both were checked against llama.cpp v0.9.0 (`cd959ba`) with `llama-tokenize --no-bos --ids`: all twelve
fixture strings tokenise to the same ids as the fixture files
`tests/fixtures/tokenize-golden-qwen2.tsv` and `tests/fixtures/tokenize-golden-llama3.tsv`. The two
models are also checked end to end against llama.cpp on the same files (greedy decoding, the chat
path and the goldens under `tests/golden/<tier>/`).

### The two chat-template files

The chat templates are checked against llama.cpp v0.9.0 on two files of TheBloke's, under exactly these
names in `~/models/`:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `thebloke-mistral-7b-instruct-v0.2.Q4_K_M.gguf` | <https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf> | 4368439584 | `3e0039fd0273fcbebb49228943b17831aadd55cbcbf56f0af00499be2040ccf9` |
| `thebloke-llama-2-7b-chat.Q4_K_M.gguf` | <https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf> | 4081004224 | `08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa` |

The first carries the pre-update Mistral v0.2 template (no system branch), the second declares no chat
template at all: the first is served as it is, the second with `--chat-template llama2`. Both were compared
with llama.cpp v0.9.0 on the same files — the rendered prompt, its token ids and the answer.

### The two SentencePiece oracle files

`tests/tokenize_spm_real.rs` checks the SentencePiece tokenizer on the two SentencePiece families
Peregrine renders chat templates for — Llama 2 and Mistral 7B — against token ids taken from
llama.cpp's `llama-tokenize` on the same uploads. It needs both files in `~/models/`, under exactly
these names:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `Llama-2-7b-chat-hf-Q4_K_M.gguf` | <https://huggingface.co/second-state/Llama-2-7B-Chat-GGUF/resolve/main/Llama-2-7b-chat-hf-Q4_K_M.gguf> | 4081005152 | `f475bcb77959f5721548bcc5122f4d5a0eb9d94ddb5b7fdce826bc1410ae0a8b` |
| `Mistral-7B-Instruct-v0.2.Q4_K_M.gguf` | <https://huggingface.co/mradermacher/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/Mistral-7B-Instruct-v0.2.Q4_K_M.gguf> | 4368441024 | `b2d7044c02b0ce9bce2ef8f37fb5bbab0190b774515473613400627b6f49f321` |

Both hashes match the ones Hugging Face reports for those uploads. The fixture files
`tests/fixtures/tokenize-golden-llama2-spm.tsv` and `tests/fixtures/tokenize-golden-mistral-spm.tsv`
hold, for 22 strings each, the ids `llama-tokenize --ids --no-escape` produced with and without
`--no-bos`; the test compares `Tokenizer::encode` against the third column and
`Tokenizer::encode_prompt` against the fourth.

Two strings in each fixture start with the literal `<s>`, and there the two implementations differ
by one token: llama.cpp prepends the model's BOS to every prompt *unconditionally* — so a prompt that
already begins with one gets two, which llama.cpp itself warns about — while `encode_prompt` keeps
exactly one. The test states that difference explicitly rather than hiding it.

Both models are also checked end to end against llama.cpp on the same files (greedy decoding, the
chat path and the goldens under `tests/golden/<tier>/`).

The upload matters: TheBloke's `Llama-2-7B-Chat-GGUF` and `Mistral-7B-Instruct-v0.2-GGUF` carry no
chat template and a template Peregrine does not render, and `peregrine inspect` refuses them with
"chat template is missing or not recognised" and "unsupported [INST] chat template variant". The two
files above were converted from the Hugging Face repositories with a `convert_hf_to_gguf.py` new
enough to carry the models' own templates.

### Tokenizer test vocabularies

`scripts/fetch-test-vocabs.sh` downloads the llama.cpp SPM test vocabulary (`models/ggml-vocab-llama-spm.gguf`)
and reference test vectors (`.inp` and `.out`) from pinned llama.cpp release files into `models/vocab/` (git-ignored),
validating pinned SHA-256 checksums. `crates/peregrine-core/tests/tokenize_spm.rs` tests these reference vectors
and skips with an informative message if the files are absent.

### Gemma 3 1B Instruct

Gemma 3 1B is checked end to end against llama.cpp v0.9.0 on the bartowski Q4_K_M upload:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `google_gemma-3-1b-it-Q4_K_M.gguf` | <https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q4_K_M.gguf> | 806058496 | `12bf0fff8815d5f73a3c9b586bd8fee8e7b248c935de70dec367679873d0f29d` |

The tokenizer was checked across 22 test strings against `llama-tokenize --ids` and `llama-tokenize --ids --no-bos` (100% agreement, with exactly one BOS on the prompt path).
Greedy decoding (32 tokens, 6 threads) was checked against llama.cpp on three prompts, including a 713-token prompt that exercises the 512-token sliding window: all 32/32 tokens match llama.cpp bit for bit on all three prompts.
The chat path (`/apply-template` and `/v1/chat/completions`) matches byte for byte with and without a system message (folded into the first user turn).
The model is recorded in `crates/peregrine-models/tests/golden/<tier>/` and checked in `docs/targets.md` (KL divergence at `-c 512` and `-c 1024`).

### Qwen 3 MoE (Qwen3-30B-A3B)

Qwen3-30B-A3B is checked end to end against llama.cpp v0.9.0 on the unsloth Q2_K upload:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `Qwen3-30B-A3B-Q2_K.gguf` | <https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q2_K.gguf> | 11258610240 | `db3ce897ccc9e7d9dbf17fe083cae7880a2092aa473b45eba8b77715aa9ca170` |

The tokenizer was checked across 22 test strings against `llama-tokenize --ids --no-bos` (100% agreement, 22/22 matches).
Greedy decoding (32 tokens, 8 threads) was checked against llama.cpp on three prompts: prompt 2 and prompt 3 match bit for bit across all 32 steps (32/32), while prompt 1 matches through the first 26 tokens with the first divergence at token 26 being a near-tie between " Portugal" (Peregrine top-1, logit 31.92) and " the" (llama.cpp top-1, logit 31.75), a margin of 0.169.
The chat path (`/apply-template` and `/v1/chat/completions`) renders identical prompt templates (15 tokens without system, 26 tokens with system) and yields matching answers identifying Paris with reasoning tokens.
The model is recorded in `crates/peregrine-models/tests/golden/<tier>/` and checked in `docs/targets.md` (KL divergence at `-c 512`, 20 chunks).

### Plain Qwen 3.5 (Qwen3.5-2B)

Qwen3.5-2B is checked end to end against llama.cpp v0.9.0 on the Q4_K_M file:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `Qwen3.5-2B-Q4_K_M.gguf` | `~/models/Qwen3.5-2B-Q4_K_M.gguf` | 1280835840 | `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223` |

The file carries no `prism.hadamard.*` metadata and is verified by `peregrine inspect`.
The tokenizer was checked across 22 test strings against `llama-tokenize --ids --no-bos` (100% agreement, 22/22 matches).
Greedy decoding (32 tokens, 8 threads) was checked against llama.cpp across four prompts, including a 602-token prompt from wikitext-2: prompts 1, 2, and 4 match bit for bit across all 32 steps (32/32), while prompt 3 matches through the first 27 tokens with the first divergence at token 27 being a near-tie between " book" (Peregrine top-1, logit 14.92) and " printing" (llama.cpp top-1, logit 14.97, logprob margin 0.045).
The chat path (`/v1/chat/completions`) generates matching reasoning tokens with answers identifying Paris as the capital of France.
The model is recorded in `crates/peregrine-models/tests/golden/<tier>/` and checked in `docs/targets.md` (KL divergence at `-c 512`, 20 chunks).

### Phi-4 (Phi 4, 14B)

Phi-4 is checked end to end against llama.cpp v0.9.0 on the Q4_K_M file:

| File | Download | Bytes | SHA-256 |
|---|---|---|---|
| `phi-4-Q4_K_M.gguf` | `~/models/phi-4-Q4_K_M.gguf` | 9053114816 | `009aba717c09d4a35890c7d35eb59d54e1dba884c7c526e7197d9c13ab5911d9` |

The file declares the `phi3` architecture (40 blocks, embedding 5120, feed-forward 17920, context 16384) and a GPT-2 byte-level BPE vocabulary whose pre-tokenizer is `dbrx`. llama.cpp v0.9.0 splits that with the Llama 3 pattern (`src/llama-vocab.cpp`, lines 310-315), so `dbrx` is a Peregrine pre-type of its own now, and the ids are checked against `llama-tokenize` in `crates/peregrine-core/tests/fixtures/tokenize-golden-phi4.tsv`: 22 of 22 strings match exactly (`llama-tokenize --ids --no-bos`, 100% agreement).
Greedy decoding (32 tokens, 8 threads) was checked against llama.cpp on three prompts: all three match bit for bit across all 32 steps (32/32 each, 96/96 steps in all). The end-of-generation tokens are taken out of the running on both sides — exactly what `llama-server`'s `ignore_eos` does — so the comparison runs the full 32 steps instead of stopping at the model's own end token.
The chat path (`/v1/chat/completions`) answers "The capital of France is Paris.", like llama.cpp's `--jinja` server, on the same prompts: 14 tokens without a system message and 24 with one. The file's template is Phi-4's own ChatML variant (`<|im_start|>role<|im_sep|>text<|im_end|>`, no newlines), which Peregrine detects by `<|im_sep|>` as llama.cpp v0.9.0 does (`src/llama-chat.cpp:99-101`, rendered at `:345-352`).
The model is recorded in `crates/peregrine-models/tests/golden/<tier>/` and checked in `docs/targets.md` (KL divergence at `-c 512`, 20 chunks).

## Accuracy against llama.cpp

`docs/dev/accuracy.md`: the KL-divergence measurement on real text (`peregrine-bench` example `kld`).

## CPU tiers

Kernels are chosen at run time from the CPU's features, capped by `PEREGRINE_ISA` (`scalar`, `avx2`,
`avx-vnni`, `avx512`; default `native`). Every feature check goes through `cpu_has!` (`crates/peregrine-kernels/src/cpu.rs`),
so a cap reaches every kernel.

```sh
PEREGRINE_ISA=scalar   cargo xtest    # portable code only
PEREGRINE_ISA=avx2     cargo xtest    # the AVX2 tier, also on an AVX-512 machine
PEREGRINE_ISA=avx-vnni cargo xtest    # the AVX-VNNI-256 tier, also on an AVX-512 machine
```

Goldens are stored per effective tier (`cpu::tier_name()`), so a forced run is compared with that tier's
goldens. Two machines must agree on a shared tier: the AVX-512 host forced to `avx2` reproduces the AVX2
host's goldens. The `x86_64-avx2-vnni256` tier aliases directly to `x86_64-avx2` goldens to check bit-identity.

CI runs the native tier on x86-64 Linux, aarch64 Linux and aarch64 macOS, and the `scalar` and `avx2` tiers
on x86-64 Linux.

GitHub's x86-64 runners have AVX2 but no AVX-512 or AVX-VNNI, so the SDE jobs run the kernel tests and the
synthetic models under Intel's Software Development Emulator:
- `sde64 -spr` (Sapphire Rapids: AVX-512 F/BW/VL, VNNI, GFNI, VBMI) fails unless the tier reported is `x86_64-avx2-avx512-vnni-gfni-vbmi`;
- `sde64 -adl` (Alder Lake: AVX-VNNI 256-bit, no AVX-512) fails unless the tier reported is `x86_64-avx2-vnni256`.

The jobs download SDE 10.13.1 from Intel's official mirror, check the pinned SHA-256, and cache the unpacked emulator.
SDE is free of charge for any type of use under the Intel Simplified Software License. No AVX-512 synthetic goldens are
committed yet, so the tests report `ok (no golden for tier ...)`; a follow-up records them on a real AVX-512 host.
