# S4.9l proof run, bluefin: the five sentences

On `origin/main` at `2a5e9f6`, Peregrine is the fastest engine in 55 of the 68 measured
embedding cells and in all nine decision cells, and it is behind only on the 128- and
512-token cells of the two small BERT encoders.

Where it leads, against the best equal-accuracy competitor: bge-m3 by 1.18-1.53x over
llama.cpp (including 1.35x at 2 000 tokens, which was 0.72x behind in the 2026-11-11
standings), EmbeddingGemma 300M by 1.30-1.51x, Qwen3-Embedding 0.6B by 1.25-1.51x, the new
Nemotron 3.5 Embed 1B by 1.34-1.62x, the mixed-length batch by 1.23-1.47x, and `/v1/systemone`
decisions by 1.33-1.84x.

Where it loses: MiniLM by 0.85-0.99x and bge-small by 0.85-0.97x against OpenVINO and ONNX
Runtime at 128 and 512 tokens, where the profile puts 58-72% of the pass in matmul and a further
23-25% in attention at 512 - OpenVINO's packed fp32 GEMM runs that matrix chain faster than
Peregrine's tile, and the F16/F32 and Q8_0 rows run at the same speed, so it is not the
activation quantisation; at 16 tokens both BERT models are now at or above parity (MiniLM up to
1.44x), closing last run's 0.74-0.87x.

Accuracy is equal: Q8_0 agrees with llama.cpp's Q8_0 within 3e-4 cosine and is closer to the
fp32 reference in all three BERT cases, the fp32/F16 rows are exact, and the ONNX/fastembed int8
exports (cosine 0.973-0.982) are shown but excluded from every ratio.

Rerank is not here because `feat: rerank and classify for the bert family` is not on main,
bge-m3 has no fp32 checkpoint for the Python engines, and `L2000_b32` exceeds every batch cap;
the cache-on table is repeated traffic, not a computation, and is not part of the claim.
