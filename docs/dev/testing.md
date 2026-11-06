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
   variant whose drafts are actually accepted. Each must load, give finite logits, be bit-identical across
   runs and across 1, 2 and 4 threads, decode greedily the same way twice, and match its golden for the
   current tier (`tests/golden/synthetic/<tier>/`). A `qwen3moe` file (8 experts, 2 used, F16 and Q8_0
   experts) is written there too: the mixture-of-experts family runs it — it loads, gives finite logits,
   is bit-identical across runs and across 1, 2 and 4 threads, matches its golden, and gives the same
   tokens alone and batched through `Engine` — and llama.cpp v0.9.0 loads and runs the same file. The
   same tests drive small models through `Engine`:
   prompts served alone must produce the same tokens when admitted together (a mixed prefill/decode step,
   both `Caches::Mapped` and, with `PEREGRINE_MIX=0`, `Caches::PerRow`), a shared opening must restore from
   the prefix store, and a LittleBit sidecar built in-test must give finite, deterministic logits. They
   need no model files, so they run in CI on every tier; the `x86_64-avx2` and scalar (`x86_64`) goldens
   were recorded on an Intel host and reproduced bit for bit on an AMD host.
   `tests/runtime_parity.rs` pins what the two runtimes do differently today, before ADR 0007 makes
   the engine the only one: the same prompt through `Session` and through a one-slot `Engine` (greedy:
   the same tokens; sampled: the session counts conversation turns into the sampling key and the engine
   does not), the taps a DSpark drafter needs (which must not change a logit bit), a second user arriving
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

## Accuracy against llama.cpp

`docs/dev/accuracy.md`: the KL-divergence measurement on real text (`peregrine-bench` example `kld`).

## CPU tiers

Kernels are chosen at run time from the CPU's features, capped by `PEREGRINE_ISA` (`scalar`, `avx2`,
`avx512`; default `native`). Every feature check goes through `cpu_has!` (`crates/peregrine-core/src/cpu.rs`),
so a cap reaches every kernel.

```sh
PEREGRINE_ISA=scalar cargo xtest    # portable code only
PEREGRINE_ISA=avx2   cargo xtest    # the AVX2 tier, also on an AVX-512 machine
```

Goldens are stored per effective tier (`cpu::tier_name()`), so a forced run is compared with that tier's
goldens. Two machines must agree on a shared tier: the AVX-512 host forced to `avx2` reproduces the AVX2
host's goldens.

CI runs the native tier on x86-64 Linux, aarch64 Linux and aarch64 macOS, and the `scalar` and `avx2` tiers
on x86-64 Linux.

GitHub's x86-64 runners have AVX2 but no AVX-512, so the `sde` job runs the kernel tests and the synthetic
models under Intel's Software Development Emulator with `sde64 -spr` (Sapphire Rapids: AVX-512 F/BW/VL,
VNNI, GFNI, VBMI). It downloads SDE 10.13.1 from Intel's official mirror, checks the pinned SHA-256, caches
the unpacked emulator, and fails unless the tier reported by the synthetic tests is
`x86_64-avx2-avx512-vnni-gfni-vbmi`. SDE is free of charge for any type of use under the Intel Simplified
Software License. No AVX-512 synthetic goldens are committed yet, so the tests report `ok (no golden for
tier ...)`; a follow-up records them on a real AVX-512 host.
