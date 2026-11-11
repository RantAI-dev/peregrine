// The fork's k-quant kernel through the fork's own library, at a real layer's shape.
//
// This is the probe S5.1's first attempt was missing: it calls the *same* entry point the
// engine's mul_mat node calls, with the *same* kinds of tensor, and it prints which of the
// fork's code paths that entry point picks for the given number of activation rows
// (`iqk_dequant_type`). That last part is the whole difference: for a repacked Q4_K matrix
// the fork's dispatcher leaves the 4-bit kernel at fewer than 32 activation rows and, at 32
// or more, converts 32 rows of weights to 8 bits and runs its 8-bit kernel instead. A probe
// that only reports GMAC/s cannot tell those two apart; this one asks.
//
// Shapes: K = 2048 (a Llama 3.2 1B q/o/gate/up row), one real matrix of 8192 rows (a gate or
// up matrix), Ny activation rows. Weights are quantised from deterministic pseudo-random
// floats with ggml's own quantiser and repacked with the fork's own in-memory repacker
// (`iqk_repack_tensor`, the thing `-rtr` calls at load). Activations go in as F32 and the
// mul_mat node converts them, exactly as in the engine.
//
// One call is K * n_out * Ny multiply-accumulates.
//
// Build against the fork's build tree (no fork source is copied into this repository):
//
//   g++ -O2 -I $FORK/ggml/include ik_kernel_probe.cpp -o ik_kernel_probe \
//       -L $FORK/build/ggml/src -lggml -Wl,-rpath,$FORK/build/ggml/src
//   taskset -c 4 ./ik_kernel_probe q4k 8192 8,64,512
//
// where FORK=~/.cache/peregrine-bench/ik_llama.cpp-479074c.

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <sys/resource.h>
#include <vector>
#include <algorithm>

extern "C" {
void iqk_repack_tensor(struct ggml_tensor * tensor);
int  iqk_repacked_type(const struct ggml_tensor * tensor);
int  iqk_dequant_type(int type, int ny);
}

static double mono_raw(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static double cpu_seconds(void)
{
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return (double)(r.ru_utime.tv_sec + r.ru_stime.tv_sec)
         + 1e-6 * (double)(r.ru_utime.tv_usec + r.ru_stime.tv_usec);
}

static const char * type_name(int t)
{
    switch (t) {
        case GGML_TYPE_Q4_K:    return "q4_k";
        case GGML_TYPE_Q4_K_R4: return "q4_k_r4";
        case GGML_TYPE_Q6_K:    return "q6_k";
        case GGML_TYPE_Q6_K_R4: return "q6_k_r4";
        case GGML_TYPE_Q8_K_R16: return "q8_k_r16";
        case GGML_TYPE_Q8_K_R8: return "q8_k_r8";
        default: return ggml_type_name((ggml_type)t);
    }
}

int main(int argc, char ** argv)
{
    const bool q6k = argc > 1 && std::string(argv[1]) == "q6k";
    const int  n_out = argc > 2 ? atoi(argv[2]) : 8192;
    const char * nys = argc > 3 ? argv[3] : "8,64,512";
    const int  reps  = argc > 4 ? atoi(argv[4]) : 5;
    const int  k     = 2048;

    struct ggml_init_params ip = { (size_t)1024 << 20, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);

    const enum ggml_type wtype = q6k ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K;
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, k, n_out);

    // Weights: deterministic pseudo-random floats, quantised by ggml itself, one row at a time.
    unsigned long long s = 0x9E3779B97F4A7C15ULL;
    std::vector<float> row(k);
    const size_t row_size = ggml_row_size(wtype, k);
    for (int r = 0; r < n_out; r++) {
        for (int i = 0; i < k; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            row[i] = (float)((double)(s >> 32) / 4294967295.0 * 2.0 - 1.0) * 0.05f;
        }
        size_t done = ggml_quantize_chunk(wtype, row.data(),
                                          (char *)w->data + (size_t)r * row_size,
                                          0, 1, k, nullptr, nullptr);
        if (done != row_size) { fprintf(stderr, "quantise mismatch\n"); return 1; }
    }

    double t0 = mono_raw();
    iqk_repack_tensor(w);
    double repack_s = mono_raw() - t0;
    printf("weights %s -> %s, %d x %d, repack (what -rtr does at load) %.3f s\n",
           type_name(wtype), type_name((int)w->type), k, n_out, repack_s);

    std::vector<std::string> ny_list;
    {
        std::string cur;
        for (const char * p = nys; ; p++) {
            if (*p == ',' || *p == 0) { ny_list.push_back(cur); cur.clear(); if (!*p) break; }
            else cur.push_back(*p);
        }
    }

    for (const auto & ny_s : ny_list) {
        const int ny = atoi(ny_s.c_str());
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, ny);
        float * af = (float *)a->data;
        for (int i = 0; i < k * ny; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            af[i] = (float)((double)(s >> 32) / 4294967295.0 * 2.0 - 1.0);
        }
        struct ggml_tensor * out = ggml_mul_mat(ctx, w, a);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 1);   // warm

        std::vector<double> us;
        for (int r = 0; r < reps; r++) {
            double r0 = mono_raw(), c0 = cpu_seconds();
            ggml_graph_compute_with_ctx(ctx, gf, 1);
            double c1 = cpu_seconds(), r1 = mono_raw();
            (void)c0; (void)c1;
            us.push_back((r1 - r0) * 1e6);
        }
        std::sort(us.begin(), us.end());
        const double macs = (double)k * n_out * ny;
        const int path = iqk_dequant_type((int)w->type, ny);
        printf("%s ny=%3d  path=%-9s  %10.1f us/call  %7.1f GMAC/s  (median of %d, min %.1f)\n",
               q6k ? "q6k" : "q4k", ny, type_name(path), us[us.size() / 2],
               macs / (us[us.size() / 2] * 1e-6) / 1e9, reps, us[0]);
    }
    ggml_free(ctx);
    return 0;
}
