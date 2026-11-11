# S5.1b: the Q4_K kernel's instruction mix

Raw measurements for `docs/design/s5-prefill-avx512.md` §9, 2026-11-12 on the desktop
(AMD Ryzen AI 9 HX 370, WSL2), one core (`taskset -c 4`).

- `peak.c` — the byte-dot roof: 16 named accumulators, one barrier per iteration, `vpdpbusd`
  zmm and ymm and `vfmadd512`. Build with
  `gcc -O2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mfma peak.c -o peak`.
- `roof.txt` — its output, twice in one run.
- `kqbench-l1.txt`, `kqbench-dram.txt` — `peregrine-bench`'s `kqbench` example, n_sb 8 (L1 tile)
  and n_sb 512 (a DRAM-sized tile), one thread: the repacked tile kernels against the row kernel.
- `ik-gemm-harness.c` — the fork's kernel through its own `ggml_mul_mat`: Q4_K_R4 or Q6_K_R4
  weights (1024 output rows) against Q8_K32 activations, one thread, 16/64/128 activation rows.
  Build against the fork's build tree:

      gcc -O2 -I ~/.cache/peregrine-bench/ik_llama.cpp-479074c/ggml/include \
          ik-gemm-harness.c -o harness \
          -L ~/.cache/peregrine-bench/ik_llama.cpp-479074c/build/ggml/src -lggml \
          -Wl,-rpath,$HOME/.cache/peregrine-bench/ik_llama.cpp-479074c/build/ggml/src
      ./harness 128 20 1024 q4k

- `ik-kernels.txt` — its output for both kernel types. It includes the activation conversion and
  the graph's per-call cost, so it is a lower bound on the kernel's own rate; at 128 activation
  rows and 1024 output rows that cost is a few percent.
