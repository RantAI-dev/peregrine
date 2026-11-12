/* The fork's 8-bit form for the rows Peregrine converted (S5.4a conformance).
 *
 * ik_llama.cpp `479074c` (`~/.cache/peregrine-bench/ik_llama.cpp-479074c`) exports
 * `iqk_convert_kquants_q8X_r8` (a C++ symbol: pass `.c` and link with `g++`, or
 * compile this file with `g++ -x c++`), which for a repacked `GGML_TYPE_Q4_K_R4`
 * matrix calls `iqk_convert_q4_k_r4_q8_k_r16` — the requantisation
 * `docs/design/s5.4-fast-prompt.md` §2 is about — and for `GGML_TYPE_Q6_K_R4`
 * calls `iqk_convert_q6_k_r4_q8_k_r16`.
 *
 * The matrix has to be in the fork's own four-row-interleaved layout, and its
 * repack routine is `static`. The repack below is a transcription of
 * `repack_q4_k` / `repack_q6_k` and `convert_q4_k` / `convert_q6_k`
 * (`ggml/src/iqk/iqk_quantize.cpp:7279-7332`, `:7387-7433`); every repacked row is
 * then compared against the file's own blocks through the fork's own
 * `dequantize_row_q4_k_r4` / `dequantize_row_q4_K` (and the Q6_K pair), so a mistake
 * in the transcription would show up as a dequantisation difference before the
 * conversion is called.
 *
 * Usage:
 *
 *   g++ -O2 -x c++ forkconv.c -o forkconv \
 *       ~/.cache/peregrine-bench/ik_llama.cpp-479074c/build/ggml/src/libggml.so -Wl,-rpath,...
 *   taskset -c 20 ./forkconv <prefix>
 *
 * Reads <prefix>-rows.bin, writes <prefix>-fork.bin in the canonical layout
 * `examples/kq8model.rs` documents ("KQ8C" | version | fmt | n_sb | n_groups, then
 * per group, per row (0..16), per super-block: f32 scale and 256 signed codes).
 *
 * IT IS NOT A WORKING ORACLE, and that is the finding this file records. In this
 * machine's cached build (`build/ggml/src/libggml.so`, built 2026-11-11 19:57 from
 * this checkout) the exported entry point returns `true` and writes zeros: for the
 * 1B model's own rows, and for a hand-built four-row R4 block with `d = 1.0`,
 * `dmin = 0`, every six-bit scale 63, every code 15 — a block whose only possible
 * output is 945/-127 codes — the scale array and the codes both come back zero.
 * The repack below is checked against the fork's own dequantiser before the call,
 * so the input is the fork's layout; the zero output is therefore the library's,
 * not this harness's. The conformance comparison in this directory is made with
 * `fork_requant.py` instead (the same arithmetic, transcribed from the same source
 * lines, with a correctly-rounded `fmaf`): see README.md.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

typedef unsigned short ggml_half;
#define QK_K 256

struct block_q4_K {
    ggml_half d;
    ggml_half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct block_q4_k_r4 {
    ggml_half d[8];
    uint8_t scales_h[16];
    uint8_t scales_l[32];
    uint8_t qs[512];
};
struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    ggml_half d;
};
struct block_q6_k_r4 {
    ggml_half d[4];
    int8_t scales[64];
    uint8_t qh[256];
    uint8_t ql[512];
};
struct block_q8_k_r16 {
    float d[16];
    int8_t qs[16 * QK_K];
};

static_assert(sizeof(block_q4_K) == 144, "q4_K");
static_assert(sizeof(block_q4_k_r4) == 576, "q4_k_r4");
static_assert(sizeof(block_q6_K) == 210, "q6_K");
static_assert(sizeof(block_q6_k_r4) == 840, "q6_k_r4");
static_assert(sizeof(block_q8_k_r16) == 64 + 16 * QK_K, "q8_k_r16");

/* The conversion is declared in the fork's C++ header without `extern "C"`, so it
 * has C++ linkage and the declaration must stay outside the block below. */
bool iqk_convert_kquants_q8X_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);

extern "C" {
const char * ggml_type_name(int type);
void dequantize_row_q4_k_r4(const block_q4_k_r4 * x, float * y, int64_t k);
void dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k);
void dequantize_row_q6_k_r4(const block_q6_k_r4 * x, float * y, int64_t k);
void dequantize_row_q6_K(const block_q6_K * x, float * y, int64_t k);
}

/* ggml's type ids for the repacked types we ask for. */
static int ggml_type_q4_k_r4(void);
static int ggml_type_q6_k_r4(void);

static void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static void convert_q4_k(const block_q4_K & x, uint8_t * L, uint8_t * Ld, uint8_t * Lm) {
    for (int ib64 = 0; ib64 < QK_K / 64; ++ib64) {
        get_scale_min_k4(2 * ib64 + 0, x.scales, Ld + 2 * ib64 + 0, Lm + 2 * ib64 + 0);
        get_scale_min_k4(2 * ib64 + 1, x.scales, Ld + 2 * ib64 + 1, Lm + 2 * ib64 + 1);
        for (int j = 0; j < 32; ++j) {
            L[64 * ib64 + j + 0] = x.qs[32 * ib64 + j] & 0xf;
            L[64 * ib64 + j + 32] = x.qs[32 * ib64 + j] >> 4;
        }
    }
}

static void convert_q6_k(const block_q6_K & x, uint8_t * L) {
    const uint8_t * ql = x.ql;
    const uint8_t * qh = x.qh;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            L[n + l + 0] = (ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            L[n + l + 32] = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            L[n + l + 64] = (ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4);
            L[n + l + 96] = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
        }
        ql += 64;
        qh += 32;
    }
}

/* `repack_q4_k`, one group of four rows and `nblock` super-blocks. */
static void repack_q4_k(const block_q4_K * x, int nblock, block_q4_k_r4 * y) {
    uint8_t L[QK_K], Ld[QK_K / 32], Lm[QK_K / 32];
    const block_q4_K * x4[4];
    for (int k = 0; k < 4; ++k) x4[k] = x + nblock * k;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        memset(y[ibl].scales_l, 0, QK_K / 8);
        memset(y[ibl].scales_h, 0, QK_K / 16);
        for (int k = 0; k < 4; ++k) {
            y[ibl].d[k + 0] = x4[k][ibl].d;
            y[ibl].d[k + 4] = x4[k][ibl].dmin;
            convert_q4_k(x4[k][ibl], L, Ld, Lm);
            for (int ib = 0; ib < QK_K / 32; ++ib) {
                y[ibl].scales_l[4 * ib + k] = (Ld[ib] & 0xf) | ((Lm[ib] & 0xf) << 4);
                uint8_t h = (Ld[ib] >> 4) | ((Lm[ib] >> 4) << 2);
                y[ibl].scales_h[(4 * ib + k) % 16] |= (h << 4 * ((4 * ib + k) / 16));
                for (int i = 0; i < 4; ++i) {
                    y[ibl].qs[64 * ib + 4 * k + i + 0] = L[32 * ib + i + 0] | (L[32 * ib + i + 8] << 4);
                    y[ibl].qs[64 * ib + 4 * k + i + 16] = L[32 * ib + i + 16] | (L[32 * ib + i + 24] << 4);
                    y[ibl].qs[64 * ib + 4 * k + i + 32] = L[32 * ib + i + 4] | (L[32 * ib + i + 12] << 4);
                    y[ibl].qs[64 * ib + 4 * k + i + 48] = L[32 * ib + i + 20] | (L[32 * ib + i + 28] << 4);
                }
            }
        }
    }
}

/* `repack_q6_k`. */
static void repack_q6_k(const block_q6_K * x, int nblock, block_q6_k_r4 * y) {
    uint8_t L[QK_K];
    const block_q6_K * x4[4];
    for (int k = 0; k < 4; ++k) x4[k] = x + nblock * k;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            y[ibl].d[k] = x4[k][ibl].d;
            convert_q6_k(x4[k][ibl], L);
            for (int ib = 0; ib < QK_K / 32; ++ib) {
                y[ibl].scales[8 * ib + k + 0] = x4[k][ibl].scales[2 * ib + 0];
                y[ibl].scales[8 * ib + k + 4] = x4[k][ibl].scales[2 * ib + 1];
                for (int i = 0; i < 4; ++i) {
                    y[ibl].ql[64 * ib + 4 * k + i + 0] = (L[32 * ib + i + 0] & 0xf) | ((L[32 * ib + i + 8] & 0xf) << 4);
                    y[ibl].ql[64 * ib + 4 * k + i + 16] = (L[32 * ib + i + 16] & 0xf) | ((L[32 * ib + i + 24] & 0xf) << 4);
                    y[ibl].ql[64 * ib + 4 * k + i + 32] = (L[32 * ib + i + 4] & 0xf) | ((L[32 * ib + i + 12] & 0xf) << 4);
                    y[ibl].ql[64 * ib + 4 * k + i + 48] = (L[32 * ib + i + 20] & 0xf) | ((L[32 * ib + i + 28] & 0xf) << 4);
                    y[ibl].qh[32 * ib + 4 * k + i + 0] = (L[32 * ib + i + 0] >> 4) | ((L[32 * ib + i + 8] >> 4) << 2) | ((L[32 * ib + i + 4] >> 4) << 4) | ((L[32 * ib + i + 12] >> 4) << 6);
                    y[ibl].qh[32 * ib + 4 * k + i + 16] = (L[32 * ib + i + 16] >> 4) | ((L[32 * ib + i + 24] >> 4) << 2) | ((L[32 * ib + i + 20] >> 4) << 4) | ((L[32 * ib + i + 28] >> 4) << 6);
                }
            }
        }
    }
}

static uint32_t rd32(const unsigned char * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: forkconv <prefix>\n");
        return 2;
    }
    std::string prefix = argv[1];
    std::string in_path = prefix + "-rows.bin";
    FILE * f = fopen(in_path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", in_path.c_str());
        return 2;
    }
    unsigned char hdr[20];
    if (fread(hdr, 1, 20, f) != 20 || memcmp(hdr, "KQ8R", 4) != 0) {
        fprintf(stderr, "%s is not a rows file\n", in_path.c_str());
        return 2;
    }
    int fmt = (int)rd32(hdr + 4);
    fprintf(stderr, "fork says 212 = %s, 214 = %s, 12 = %s, 14 = %s\n",
            ggml_type_name(212), ggml_type_name(214), ggml_type_name(12), ggml_type_name(14));
    int n_sb = (int)rd32(hdr + 8);
    int n_groups = (int)rd32(hdr + 12);
    int row_bytes = (int)rd32(hdr + 16);
    int block = fmt == 0 ? 144 : 210;
    if (row_bytes != n_sb * block) {
        fprintf(stderr, "row_bytes %d is not n_sb %d x %d\n", row_bytes, n_sb, block);
        return 2;
    }
    std::vector<unsigned char> rows((size_t)16 * row_bytes);
    std::string out_path = prefix + "-fork.bin";
    FILE * o = fopen(out_path.c_str(), "wb");
    if (!o) {
        fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return 2;
    }
    fwrite("KQ8C", 1, 4, o);
    uint32_t ver = 1, ufmt = (uint32_t)fmt, unsb = (uint32_t)n_sb, ung = (uint32_t)n_groups;
    fwrite(&ver, 4, 1, o);
    fwrite(&ufmt, 4, 1, o);
    fwrite(&unsb, 4, 1, o);
    fwrite(&ung, 4, 1, o);

    size_t weights = 0;
    const void * r4d = 0;
    for (int g = 0; g < n_groups; ++g) {
        if (fread(rows.data(), 1, rows.size(), f) != rows.size()) {
            fprintf(stderr, "short read at group %d\n", g);
            return 2;
        }
        std::vector<float> deq(4 * (size_t)n_sb * QK_K), deq_ref(4 * (size_t)n_sb * QK_K);
        std::vector<block_q8_k_r16> y(n_sb);
        memset(y.data(), 0xAA, y.size() * sizeof(block_q8_k_r16));
        if (fmt == 0) {
            std::vector<block_q4_k_r4> r4((size_t)4 * n_sb);
            for (int q = 0; q < 4; ++q) {
                const block_q4_K * src = (const block_q4_K *)(rows.data() + (size_t)(4 * q) * row_bytes);
                repack_q4_k(src, n_sb, r4.data() + (size_t)q * n_sb);
                /* The transcription's own check: the fork's dequantiser on both
                 * layouts, row by row. Rows 0..3 of the R4 group are rows 4q..4q+3
                 * of the file's. */
                std::vector<float> want(4 * (size_t)n_sb * QK_K);
                for (int r = 0; r < 4; ++r) {
                    dequantize_row_q4_K(src + (size_t)r * n_sb, deq_ref.data(), n_sb * QK_K);
                    memcpy(want.data() + (size_t)r * n_sb * QK_K, deq_ref.data(),
                           sizeof(float) * n_sb * QK_K);
                }
                dequantize_row_q4_k_r4(r4.data() + (size_t)q * n_sb, deq.data(), 4 * n_sb * QK_K);
                if (memcmp(want.data(), deq.data(), sizeof(float) * want.size()) != 0) {
                    fprintf(stderr, "the Q4_K repack differs from the file's weights (group %d, rows %d..)\n", g, 4 * q);
                    return 1;
                }
            }
            /* bx is the row stride of the repacked type: four rows share n_sb blocks. */
            r4d = r4.data();
            size_t bx = (size_t)n_sb * sizeof(block_q4_k_r4) / 4;
            bool ok = iqk_convert_kquants_q8X_r8(ggml_type_q4_k_r4(), n_sb * QK_K, r4.data(), bx, y.data(), 16);
            fprintf(stderr, "q4 convert returned %d\n", (int)ok);
            if (!ok) {
                fprintf(stderr, "the fork refused the Q4_K conversion\n");
                return 1;
            }
        } else {
            std::vector<block_q6_k_r4> r4((size_t)4 * n_sb);
            for (int q = 0; q < 4; ++q) {
                const block_q6_K * src = (const block_q6_K *)(rows.data() + (size_t)(4 * q) * row_bytes);
                repack_q6_k(src, n_sb, r4.data() + (size_t)q * n_sb);
                std::vector<float> want(4 * (size_t)n_sb * QK_K);
                for (int r = 0; r < 4; ++r) {
                    dequantize_row_q6_K(src + (size_t)r * n_sb, deq_ref.data(), n_sb * QK_K);
                    memcpy(want.data() + (size_t)r * n_sb * QK_K, deq_ref.data(),
                           sizeof(float) * n_sb * QK_K);
                }
                dequantize_row_q6_k_r4(r4.data() + (size_t)q * n_sb, deq.data(), 4 * n_sb * QK_K);
                if (memcmp(want.data(), deq.data(), sizeof(float) * want.size()) != 0) {
                    fprintf(stderr, "the Q6_K repack differs from the file's weights (group %d, rows %d..)\n", g, 4 * q);
                    return 1;
                }
            }
            r4d = r4.data();
            size_t bx = (size_t)n_sb * sizeof(block_q6_k_r4) / 4;
            bool ok = iqk_convert_kquants_q8X_r8(ggml_type_q6_k_r4(), n_sb * QK_K, r4.data(), bx, y.data(), 16);
            fprintf(stderr, "q6 convert returned %d\n", (int)ok);
            if (!ok) {
                fprintf(stderr, "the fork refused the Q6_K conversion\n");
                return 1;
            }
        }
        if (getenv("FORKCONV_DEBUG")) {
            const unsigned char * raw = (const unsigned char *)y.data();
            fprintf(stderr, "input r4 d (f16 hex) rows 0..3:");
            for (int r = 0; r < 4; ++r) {
                ggml_half h = (fmt == 0) ? ((block_q4_k_r4 *)r4d)->d[r]
                                         : ((block_q6_k_r4 *)r4d)->d[r];
                fprintf(stderr, " %04x", h);
            }
            fprintf(stderr, "\nall 16 y d:");
            for (int i = 0; i < 16; ++i) fprintf(stderr, " %g", y[0].d[i]);
            fprintf(stderr, "\n");
            fprintf(stderr, "d as floats:");
            for (int i = 0; i < 4; ++i) fprintf(stderr, " %g", y[0].d[i]);
            fprintf(stderr, "\nraw bytes[0..8]:");
            for (int i = 0; i < 8; ++i) fprintf(stderr, " %02x", raw[i]);
            fprintf(stderr, "\nfile d of rows 0..3:");
            if (fmt == 0) {
                for (int r = 0; r < 4; ++r) {
                    ggml_half h;
                    memcpy(&h, rows.data() + (size_t)r * row_bytes, 2);
                    fprintf(stderr, " %04x", h);
                }
            } else {
                for (int r = 0; r < 4; ++r) {
                    ggml_half h;
                    memcpy(&h, rows.data() + (size_t)r * row_bytes + 208, 2);
                    fprintf(stderr, " %04x", h);
                }
            }
            fprintf(stderr, "\nqs[0]=%d qs[32]=%d qs[64]=%d qs[512]=%d\n",
                    y[0].qs[0], y[0].qs[32], y[0].qs[64], y[0].qs[512]);
        }
        /* The canonical order: row, super-block, scale, 256 signed codes. */
        for (int k = 0; k < 16; ++k) {
            for (int sb = 0; sb < n_sb; ++sb) {
                float d = y[sb].d[k];
                fwrite(&d, 4, 1, o);
                signed char codes[QK_K];
                for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                    for (int l = 0; l < 8; ++l) {
                        for (int j = 0; j < 4; ++j) {
                            unsigned char u = (unsigned char)y[sb].qs[512 * ib32 + 64 * l + 4 * k + j];
                            codes[32 * ib32 + 4 * l + j] = (signed char)(u ^ 0x80);
                        }
                    }
                }
                fwrite(codes, 1, QK_K, o);
                weights += QK_K;
            }
        }
    }
    fclose(o);
    fclose(f);
    printf("%s: %d groups of 16 rows, %d super-blocks, %zu weights converted by the fork\n",
           out_path.c_str(), n_groups, n_sb, weights);
    return 0;
}

/* The ids from the fork's `ggml.h`: `GGML_TYPE_Q4_K_R4 = 212`,
 * `GGML_TYPE_Q6_K_R4 = 214` (the plain types are 12 and 14). */
static int ggml_type_q4_k_r4(void) { return 212; }
static int ggml_type_q6_k_r4(void) { return 214; }
