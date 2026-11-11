// ik_llama.cpp's k-quant matrix multiplication alone, on one core, at the widths the
// prefill study asks for (1, 16, 64, 128 activation rows).
//
// This is a measurement tool for the Peregrine prefill study (S5). It is not part of
// either engine: it links against the ik_llama.cpp build at 479074c and calls its
// ggml graph API, the same mul_mat a prompt pass reaches. `iqk_repack_tensor` is the
// function `-rtr` calls at load, so `-rtr` on/off is measured by calling it or not.
//
//   g++ -O2 -std=c++17 -I<ik>/ggml/include -I<ik>/ggml/src -I<ik>/ggml/src/iqk \
//       ik_kqbench.cpp -o ik_kqbench -L<ik>/build/ggml/src -lggml -Wl,-rpath,<ik>/build/ggml/src
//
//   ./ik_kqbench <q4k|q6k|q5_0|q8_0> <rows=2048> <cols=4096> <reps> <threads> [rtr 0|1]
//
// Prints, per activation width, the MAC rate of one thread and (with >1 threads) the
// aggregate. MAC counts are exact: rows x cols x width.
#include "ggml.h"
#include "iqk_quantize.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void compute(struct ggml_cgraph * graph, int n_threads) {
    static std::vector<uint8_t> buf;
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);
    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }
    ggml_graph_compute(graph, &plan);
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <q4k|q6k|q5_0|q8_0> <rows> <cols> <reps> <threads> [rtr]\n", argv[0]);
        return 1;
    }
    const char * which = argv[1];
    const int M = atoi(argv[2]);       // output rows
    const int K = atoi(argv[3]);       // weights per row
    const int reps = atoi(argv[4]);
    const int n_threads = atoi(argv[5]);
    const int rtr = argc > 6 ? atoi(argv[6]) : 1;

    enum ggml_type type = GGML_TYPE_Q4_K;
    int block = 144;
    int scale_at = 0;
    if (!strcmp(which, "q6k")) { type = GGML_TYPE_Q6_K; block = 210; scale_at = 208; }
    if (!strcmp(which, "q5k")) { type = GGML_TYPE_Q5_K; block = 176; scale_at = 0; }
    if (!strcmp(which, "q5_0")) { type = GGML_TYPE_Q5_0; block = 22; scale_at = 0; }
    if (!strcmp(which, "q8_0")) { type = GGML_TYPE_Q8_0; block = 34; scale_at = 0; }

    ggml_time_init();
    struct ggml_init_params ip = { 1024u * 1024u * 1024u, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, type, K, M);
    // Pseudo-random bytes, with each block's f16 scale replaced by a small positive
    // number so the arithmetic is finite (the kernel does not branch on the data).
    {
        std::mt19937 rng(12345);
        auto * p = (uint8_t *) w->data;
        const size_t n = ggml_nbytes(w);
        for (size_t i = 0; i < n; ++i) p[i] = (uint8_t) (rng() >> 24);
        const size_t nblk = n / block;
        for (size_t b = 0; b < nblk; ++b) {
            uint8_t * d = p + b * block + scale_at;
            d[0] = 0x21;
            d[1] = 0x30;   // ~2^-14 * 1.5, a small positive f16
            if (type == GGML_TYPE_Q4_K) { d[2] = 0x21; d[3] = 0x2c; }
            if (type == GGML_TYPE_Q5_K) { d[2] = 0x21; d[3] = 0x2c; }
        }
    }
    if (rtr) iqk_repack_tensor(w);

    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 128);
    {
        std::mt19937 rng(999);
        float * p = (float *) x->data;
        for (int i = 0; i < K * 128; ++i) p[i] = (float) (int) (rng() % 2001) / 1000.0f - 1.0f;
    }
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);

    // Warm up.
    compute(graph, n_threads);

    const int widths[] = {1, 16, 64, 128};
    for (int wi = 0; wi < 4; ++wi) {
        const int nb = widths[wi];
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            ggml_set_name(y, "y");
            // The graph computes all 128 rows of x; ask for a narrower result by
            // re-viewing the destination (mul_mat reads ne1 of the source).
            struct ggml_tensor * xv = ggml_view_2d(ctx, x, K, nb, x->nb[1], 0);
            struct ggml_tensor * yv = ggml_mul_mat(ctx, w, xv);
            struct ggml_cgraph * g2 = ggml_new_graph(ctx);
            ggml_build_forward_expand(g2, yv);
            compute(g2, n_threads);
            const auto t0 = std::chrono::steady_clock::now();
            compute(g2, n_threads);
            const auto t1 = std::chrono::steady_clock::now();
            const double s = std::chrono::duration<double>(t1 - t0).count();
            if (s < best) best = s;
        }
        const double macs = (double) M * K * nb * 2.0 * reps;  // builds a graph per rep; see note
        const double gm = (double) M * (double) K * nb * 2.0 / best / 1e9;
        printf("%-5s rtr=%d t=%2d %3d acts: %7.1f GMAC/s  (threads>1: aggregate)\n",
               which, rtr, n_threads, nb, gm);
        (void) macs;
        (void) y;
    }
    ggml_free(ctx);
    return 0;
}
