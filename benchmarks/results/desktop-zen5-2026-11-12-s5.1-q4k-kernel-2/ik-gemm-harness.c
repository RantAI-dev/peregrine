// Time the fork's repacked Q4_K kernel through ggml_mul_mat: 8 output rows (one tile per
// 256 weights... GGML_TYPE_Q4_K_R4 holds four rows per block) against N activation rows.
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

int main(int argc, char **argv) {
    int n_in = 2048;      // 8 super-blocks
    int n_out = argc > 3 ? atoi(argv[3]) : 64;
    enum ggml_type ty = (argc > 4 && argv[4][1] == 0x36) ? GGML_TYPE_Q6_K_R4 : GGML_TYPE_Q4_K_R4;
    int n_act = argc > 1 ? atoi(argv[1]) : 8;
    int reps  = argc > 2 ? atoi(argv[2]) : 2000;
    struct ggml_init_params p = { (size_t)512 << 20, NULL, false };
    struct ggml_context *ctx = ggml_init(p);
    struct ggml_tensor *w = ggml_new_tensor_2d(ctx, ty, n_in, n_out);
    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_act);
    srand(1234);
    unsigned char *wb = (unsigned char *)w->data;
    for (size_t i = 0; i < ggml_nbytes(w); i++) wb[i] = (unsigned char)(rand() & 0xFF);
    float *af = (float *)a->data;
    for (int i = 0; i < n_in * n_act; i++) af[i] = (float)((rand() % 1000) - 500) / 500.0f;
    struct ggml_tensor *r = ggml_mul_mat(ctx, w, a);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    double best = 1e9;
    for (int k = 0; k < 5; k++) {
        double t0 = now();
        for (int i = 0; i < reps; i++) ggml_graph_compute_with_ctx(ctx, gf, 1);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    double macs = (double)n_out * n_in * n_act * reps;
    printf("%s act %3d out %4d: %.2f us/call  %.1f GMAC/s/core\n",
           ty == GGML_TYPE_Q6_K_R4 ? "q6k" : "q4k", n_act, n_out,
           best / reps * 1e6, macs / best / 1e9);
    ggml_free(ctx);
    return 0;
}
