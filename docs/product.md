# Product

## One line

**Peregrine is the fastest way to run small and mid-size language models on a CPU you already have,
with numbers anyone can check.**

## The problem

Most people and most companies who want a private language model do not have a GPU to spare. They have
laptops, desktops, office servers and cloud VMs. On that hardware:

- engines are tuned for GPUs first, and CPU paths are an afterthought;
- speed claims are hard to verify: different files, thread counts, contexts and machines;
- the newest efficient models (ternary weights such as Bonsai 2, hybrid recurrent layers, MTP heads)
  arrive on CPUs late or not at all.

## What Peregrine is

- **A CPU inference engine** in Rust: a library (`peregrine-core`) and an OpenAI-compatible server
  (`peregrine-server`).
- **Measured against the machine.** It reports bytes moved per token beside tokens per second, so every run
  shows how close it is to the memory-bandwidth or compute ceiling.
- **Lossless by default.** Batching, speculation and kernels give the same tokens as the exact path.
- **Built for many users at once:** one weight pass serves every active request.

## Who it is for

| User | What they need | What Peregrine gives them |
|---|---|---|
| Teams running private assistants on office servers or VMs | throughput for 5-50 users without GPUs | batching, prefix reuse, `/metrics`, OpenAI API |
| Developers on laptops | a fast local model with no setup | one binary, one model file |
| Builders of products on top (RantAI and others) | an engine they can embed and trust | a Rust library with stable invariants and one dependency |
| Researchers comparing engines | reproducible numbers | the benchmark harness, raw JSON, fraction-of-ceiling columns |

## Scope

**In scope**
- Decoder-only language models up to about 30B total parameters (MoE included), the size class that runs
  usefully on CPUs.
- GGUF as the model format, including the low-bit and ternary formats.
- x86-64 (AVX2 and newer) first; ARM (NEON/SDOT) as a second target.
- Chat completion serving: streaming, many users, speculation, prefix caching.

**Out of scope, on purpose**
- GPUs and NPUs. Other engines do this well; Peregrine's value is the CPU.
- Training and fine-tuning.
- Formats other than GGUF (ONNX, OpenVINO IR, MLX). Conversion tools exist.
- Lossy speedups turned on by default.

**Later, not now**
- Vision and audio inputs (the Qwen 3.5 vision tower is the first candidate).
- Embedding and reranking endpoints.

## Claims we intend to make

Each becomes a claim only after the benchmark harness shows it on a clean machine (`docs/targets.md`):

1. **Fastest CPU engine for ternary models on x86** (Bonsai 2 and its formats).
2. **Matches or beats llama.cpp and ik_llama.cpp on the popular standard GGUF models**, and wins on many
   users and on MTP-capable models.

We will not claim "fastest CPU engine" without a scope. Every published table names the model, format,
CPU and measure, and shows the cases where we lose.

## Principles

1. **Measure before optimising, and publish the method.**
2. **Never trade correctness for speed by default.**
3. **Own the hot path** (kernels, threading, cache, sampling); use well-tested crates for everything else.
4. **Breadth through structure:** adding a format, an architecture or a CPU family touches one place.
5. **Name the competition precisely:** every comparison names the
   competitor's version.
