# peregrine-core

The Peregrine engine as a library: GGUF and safetensors loading, tokenizer and chat templates, CPU kernels,
the thread pool, the cache, sampling, speculation and the batching scheduler (`serve::Engine`). One runtime
dependency (`memmap2`). It knows nothing about HTTP; see `peregrine-server`.

A map of the modules is in [`ARCHITECTURE.md`](../../ARCHITECTURE.md); what runs today is in the
[README](../../README.md#what-runs-today).

## Highlights

- **One weight pass for many users.** `Model::forward_rows` takes a position and a cache per row, so every
  active user advances in one read of the weights. Attention runs one row at a time per user, so a user's
  logits equal a solo run's bit for bit.
- **Position-keyed sampling** (`sample.rs`): the draw depends only on (seed, turn, position), which makes
  batching and speculation reproducible at any temperature. Top-k, top-p and min-p follow llama.cpp's order.
- **Caches** (`kv.rs`, `prefix.rs`): one type for attention K/V and recurrent state, an optional
  Hadamard-rotated int8 KV cache, snapshots for speculation, and a prefix store that skips prefill for a
  shared prompt beginning.
- **Speculation:** MTP heads, DSpark block drafting and prompt lookup, all verified losslessly.
- **LittleBit** (`lbq.rs`, arXiv 2506.13771): 1-bit-class weights, off by default.

Tests that need model files skip when the files are absent; see `docs/dev/testing.md`.
