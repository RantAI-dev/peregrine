# Accuracy against llama.cpp: the KL-divergence measurement

`crates/peregrine-bench/examples/kld.rs` compares Peregrine's next-token distributions with llama.cpp's on
real text, the way `llama-perplexity --kl-divergence` compares two llama.cpp runs.

## What it measures, and what it does not

llama.cpp writes a reference file with `--kl-divergence-base`: the evaluation tokens and, for every scored
position, its log-probabilities over the whole vocabulary. `kld` runs the same tokens through Peregrine with the
same chunking (chunks of `n_ctx` tokens, the second half of each chunk scored) and reports, per position, the
KL divergence of Peregrine's distribution from llama.cpp's, the difference in the probability of the correct
token (Δp), and whether both engines put the same token on top.

It measures the **distance between two engines** that both approximate the exact computation: both quantise
activations (llama.cpp to Q8_K for k-quant matrices, Peregrine to its own int8 layouts) and sum in different
orders. It is not the error against the exact computation. The accuracy budget in `docs/targets.md` (mean KL
≤ 1e-4) is against Peregrine's own exact f32 path and is a separate measurement.

Sanity check: llama.cpp compared against its own base file gives a mean KLD of 0.000000, so the base file and
the comparison are consistent.

## Like-for-like: the same cache precision on both sides

The measurement above compares two engines that do the same computation *differently*. Peregrine keeps its
queries in f32 but rounds its cached keys and values to f16; llama.cpp's cache is f16 by default, so its
attention reads f16 operands as well. A difference between the two is then a difference between two *different*
computations, and it is not obvious which engine it comes from.

For a comparison in which both engines' attention reads f32 keys and values, run llama.cpp with a full-precision
cache and Peregrine with the f32 KV cache (`KvPrec::F32`):

```sh
llama-perplexity -m model.gguf -f wiki.test.raw -c 512 -b 512 -t 8 --chunks 40 \
    --cache-type-k f32 --cache-type-v f32 --flash-attn off \
    --kl-divergence-base model.f32.base.bin
cargo run --release -p peregrine-bench --example kld -- \
    --base model.f32.base.bin --model model.gguf --threads 8 --kv f32
```

The flags, and what each one is for:

| Flag | Side | Why |
|---|---|---|
| `--cache-type-k f32 --cache-type-v f32` | llama.cpp | the keys and values in the cache keep full precision, so its attention reads what the model produced |
| `--cache-type-k f16 --cache-type-v f16` | llama.cpp | the same run with llama.cpp's default cache: the pair of rows isolates the cache precision |
| `--flash-attn off` | llama.cpp | llama.cpp's flash-attention kernels do not read a full-precision cache, and the two are different computations (on 2 chunks of SmolLM2, PPL 13.7227 with flash attention on and 13.7081 with it off). `auto` picks flash attention *on* for the CPU, so it has to be turned off explicitly |
| `--kv f32` | Peregrine | the same precision on this side (`--kv f16`, `--kv i8`; the default is f16) |
| `PEREGRINE_KV=f32` | Peregrine | the same choice for the engine as a whole (`peregrine serve --kv f32` says it per run) |

The result is in `docs/targets.md`. It is worth stating what it says, because it is not the expected answer: the
cache precision moves each engine's *own* output by about as much as the two engines differ (llama.cpp against
itself, f16 against f32, gives a mean KLD of 0.0132 on Qwen3-1.7B — the same size as Peregrine against
llama.cpp), but it does not create the distance between them: the two engines' keys and values are nearly the
same, so they round to the same halves and both distributions move the same way.

## Reproducing

llama.cpp v0.9.0 (commit `cd959ba`), CPU build. Corpus: wikitext-2-raw `wiki.test.raw`
(<https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip>, the file llama.cpp's
`scripts/get-wikitext-2.sh` fetches; `wiki.test.raw` SHA-256
`173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`).

```sh
llama-perplexity -m model.gguf -f wiki.test.raw -c 512 -b 512 -t 8 --chunks 40 \
    --kl-divergence-base model.base.bin
cargo run --release -p peregrine-bench --example kld -- \
    --base model.base.bin --model model.gguf --threads 8
```

The base file is large (vocabulary × scored positions; 3 GB for Qwen 2.5's 152k vocabulary at 40 chunks): keep it
outside the repository and delete it afterwards.

## Reading the output

- **Mean KLD ± error**: the headline number. Below ~0.002 the two engines agree about as closely as two
  builds of the same engine with different activation quantisation.
- **Same top p**: how often both engines pick the same most likely token. Greedy decoding diverges where this
  differs, at near-ties.
- **99% / maximum KLD**: the tail. A single large maximum with a small 99% is a near-tie; a large 99% means a
  systematic difference.
- **PPL(Q) vs PPL(base)**: Peregrine's perplexity against llama.cpp's on the same tokens.

Results are recorded in `docs/targets.md` with the date, host, Peregrine commit and llama.cpp version.
