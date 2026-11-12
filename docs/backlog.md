# Backlog

Ideas that are not on the [roadmap](roadmap.md) yet. Each says what it is, why it might be worth
doing, what the evidence is, and how big it looks. Nothing here is a commitment or a measured
result: an item moves to the roadmap when someone has decided to do it, and a number becomes a
claim only after it has been measured on the harness.

How to read an entry:

- **Result**: *exact* (bit-identical to today), *same maths* (a different order of operations:
  rounding-level differences, goldens change once), *lossy* (changes what the model computes; opt-in
  with an accuracy report, ADR 0003).
- **Evidence**: *measured here*, *reported* (by a paper or another project, on their hardware),
  *estimate* (ours, unmeasured).
- **Size**: S (days), M (about a week), L (more).

Sources for the research behind this page: the competitor code study, the literature survey and
the stage profile of 2026-11-11 that `docs/design/s4.9-sota.md` is built on.

## 1. Speed: all models

| Idea | Why | Result | Evidence | Size |
|---|---|---|---|---|
| **Prompt processing on AVX-512 at ik_llama.cpp's level** | The fork processes prompts about twice as fast as Peregrine on the Zen 5 desktop (0.39-0.58×, four Q4_K_M models, 512 and 4096 context). It repacks k-quant weights at load into interleaved rows for its own AVX-512 integer kernels and has its own attention kernels; why Peregrine's AVX-512 kernels are half as fast is not known yet. First step: read the fork's kernels and profile Peregrine's prefill on the same machine, then decide what to build. This is the largest measured loss. | exact or same maths | measured here | L |
| **Larger prefill passes for chat on big float files** | A pass streams every weight once per 16-64 rows, so a 1.2 GB F16 model is close to memory-bandwidth-bound in prefill while the new float kernel sits at 90% of the FMA peak. S4.9b fixes this for embedding and decision work; chat keeps its chunk for decode fairness. A pass sized by a byte budget (weights streamed per row) rather than a row count would cover chat too. | exact | estimate; a chunk sweep is being measured | S |
| **Work split by core speed** | Laptop and desktop CPUs mix fast and slow cores (Intel P/E, Zen 5/5c, ARM big.LITTLE). Equal shares make every pass wait for the slowest core. Splitting in proportion to measured per-core speed was reported at +20-30% prefill and +9-22% decode on hybrid Intel CPUs. Peregrine's rows are claimed from a counter in places; panels are not. | exact | reported | M |
| **One join per layer for batches** | A layer issues 12-14 fork-joins. With whole texts (or whole row blocks) owned by a thread for a layer, there is one. Costs more weight traffic per thread, so it is for batches only. | exact | estimate | M |
| **Read-row-only last block** | An embedding with last-token or CLS pooling, and every decision, reads one row of the final layer. That row's attention needs every key but only one query; its feed-forward is one row. Saves most of one layer: about 1/L of the pass (15% on a 6-layer encoder), nothing published. | exact | estimate | S |
| **Packed weights shared across processes** | S4.9e caches the repacked tiles on disk. Mapping that cache instead of copying it lets several processes (a chat server, an embedder, CLI calls) share the pages and start instantly. | exact | estimate | S |
| **Huge pages for the packed store by default where available** | The store is streamed linearly; 2 MB pages cut TLB misses. It is an option today (`tile_hugepages`); measure and choose the default. | exact | to measure | S |
| **AVX-512 BF16 and AMX tiers** | Sapphire Rapids and later Xeons have AMX-INT8/BF16 tile registers; reports of up to 4× over AVX-512 bf16 for encoders. Needs a host to develop on. BF16 dot products fuse differently from an FMA chain, so this tier cannot be bit-identical to the others: it would be its own golden tier. | same maths | reported | L |
| **NEON and SVE float kernel** | S4.9f left aarch64 on the old float tile. The sequential-FMA chain is the same on NEON, so results stay identical to x86. | exact (vs x86) | estimate | M |

## 2. Speed: embeddings

| Idea | Why | Result | Evidence | Size |
|---|---|---|---|---|
| **Result cache and in-batch duplicates** | Production embedding traffic repeats itself (one deployment reported a 40% hit rate and nearly doubled throughput). A cache keyed by token ids is exact; so is computing a text once when it appears twice in a batch. | exact | reported | S |
| **Q8_0 weights with float activations** | The exact reading of a Q8_0 encoder file (decision 2 of the S4.9 design). Keeps weights four times smaller than fp32 with no activation rounding: aims at the single-short-text latency that int8 engines win today, at full accuracy. | exact for the file | estimate | M |
| **Exact banded attention for local layers at long lengths** | ModernBERT-class encoders alternate local (128-token window) and global attention. Exporters to graph runtimes compute a dense n×n matrix and mask it; a banded kernel does only the window. About 1.7-1.9× fewer multiplies at 8k tokens. | same maths | estimate | M (with S4.7) |
| **Integer activations on VNNI, opt-in** | 2.5-4× reported on VNNI and i8mm CPUs at 0.4-1.5% retrieval loss. Peregrine has the kernels; what is missing is per-row scales that keep batch invariance, and the accuracy report. | lossy | reported | M |

## 3. Speed and capability: decision models

| Idea | Why | Result | Evidence | Size |
|---|---|---|---|---|
| **State handles** | An agent loop asks many batches of questions about one state. Let a client register a state once (`state_id`) and ask against it; the engine keeps the forked snapshot. Today the prefix store finds it by matching tokens; a handle makes it explicit and pins it. | exact | estimate | S |
| **Many states, same questions** | Labelling a dataset is N states × the same questions. The questions' suffix tokens are shared too; the pass can carry all states' rows together with no padding. A dedicated batch shape for `/v1/systemone`. | exact | estimate | S |
| **Where to checkpoint a hybrid** | A snapshot of a Qwen 3.5 state is 20 MB and valid at one position. A policy for which positions to keep (request boundaries, the end of the system prompt, every k tokens of a long state) decides the hit rate; work on hybrid-model caches reports large gains from choosing well. | exact | reported (GPU) | M |
| **Decision latency class** | A guard or router answer gates a user-visible action. When a decision shares the engine with chat prefill, let it go first in the pass order. | exact | estimate | S |
| **`box` questions and assets** | The TypeSafe API has a `box` question type and `assets` that llama.cpp's endpoint does not serve; follow when a model needs them. Image input (`files`, `images`) needs the vision projector path. | n/a | spec | L |

## 4. Product

| Idea | Why | Size |
|---|---|---|
| **Sparse and multi-vector outputs for bge-m3** | bge-m3's selling point is three outputs from one pass: dense, sparse (lexical weights) and ColBERT multi-vector. llama.cpp serves only the dense one. Serving all three from one pass would be a first for a GGUF engine. | M |
| **Span pooling ("late chunking")** | Embed a long document once and return one vector per span of tokens: the chunk vectors see the whole document. It is a pooling option on a pass that already ran, so it costs almost nothing. | S |
| **Output quantisation** | `encoding_format: int8` and `binary` with the model's documented calibration, as the hosted APIs offer: 4× and 32× less storage for the caller, about 99% and 96% of retrieval quality with rescoring (reported). Costs the engine nothing. | S |
| **Offline embedding jobs** | `peregrine embed --input texts.jsonl --output vectors.npy`: no HTTP, token-budget passes, resumable. The throughput case in its purest form. | S |
| **`logprobs` and `top_logprobs` on chat** | The label read-out is most of the machinery. Asked for by evaluation tools and by people who classify with a chat model. | S |
| **Image input for Nemotron 3.5 Embed** | The model embeds page images as well as text, through a Pixtral vision encoder and projector shipped as an `mmproj` file. llama.cpp loads that file but does not accept images on its embedding endpoint for this model yet, so serving them would be a first for a GGUF engine. Needs the vision encoder, image preprocessing and an input shape on `/v1/embeddings`. | L |
| **`input_type` on `/v1/embeddings`** | Embedders are trained with a prefix per role (`query: ` and `passage: ` for Nemotron and e5, a task sentence for Qwen3-Embedding and EmbeddingGemma). Today the client writes it. An `input_type: query | document` field that applies the model's own format, as the hosted APIs offer, removes a common source of silently worse retrieval. | S |
| **Sleeping and loading models on demand** | With several models in one server (S4.10): unload idle ones, advertise them in `/v1/models` while asleep (llama.cpp's router does), reload on first use from the packed-weight cache. | M |

## 5. Proof and trust

| Idea | Why | Size |
|---|---|---|
| **Chat baselines against the other engines the targets name** | `docs/targets.md` lists ik_llama.cpp, OpenVINO GenAI and MNN beside llama.cpp, and bitnet.cpp / T-MAC for ternary models. ik_llama.cpp is now measured on the desktop (`docs/benchmark.md`): decode level, 10 users ahead, prompt processing behind at 0.39-0.58×. Still to measure: ik_llama.cpp on bluefin and ARM, OpenVINO GenAI, MNN. | M |
| **A controlled accuracy study of GGUF formats for embedders** | Nobody has published MTEB or BEIR numbers for Q8_0 / Q4_K embedding files, only cosine drift. Running NanoBEIR and four BEIR sets per format, per engine, tells users which file to download, and is publishable on its own. | M |
| **Batch-invariance as a published property** | ONNX Runtime's quantised path uses one activation scale for the whole batch, so a text's vector depends on its batch-mates; padded engines can differ by mask handling. A small public test that any engine can run ("same text, different batch: same bits?") makes Peregrine's guarantee visible. | S |
| **Cost per million embeddings** | The number buyers compare. Publish it for a small cloud VM alongside texts per second. | S |
| **A quiet benchmark host and a performance gate** | Every measurement in November fought a busy machine. A dedicated runner would allow a CI job that fails a PR on a throughput regression beyond noise. | M |

## 6. Hygiene

| Idea | Why | Size |
|---|---|---|
| **Fuzz the request parsers** | `json.rs`, the GGUF header and decision-metadata readers and the tokenizers take untrusted input in a server. `cargo fuzz` targets with a short CI run. | S |
| **A server test waits for its own server** | The server tests pick a free port by binding port 0 and releasing it, then wait for `/health` on that port. Two tests running in parallel can be handed the same port: the second server fails to bind, its test talks to the first test's server, and gets an empty reply when that test ends and kills it (seen once in CI on aarch64). Wait for the child's own "listening" line and retry on a bind failure. | S |
| **Per-test directories as a helper** | Two tests sharing a temp path caused a SIGBUS in CI. One helper that names a directory after the test, used everywhere. | S |

## Not planned

Token pruning, early exit and distilled static-embedding models change the model itself, not how it
is run; they are for model authors. Reuse of cached state that is not a true prefix (cache
blending, block attention) needs models trained for it.
