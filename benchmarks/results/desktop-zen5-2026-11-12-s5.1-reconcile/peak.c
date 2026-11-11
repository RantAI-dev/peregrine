/* The core's byte-dot roof, counted by hand.
 *
 * Every probe is a loop whose instruction count per iteration is fixed (one instruction per
 * named accumulator per iteration) so the GMAC/s can be checked two ways: MACs per
 * instruction times instructions per second, and MACs per cycle times cycles per second.
 * The core clock is measured independently by dependent one-cycle `add` chains of two
 * different unroll lengths (a loop overhead of one cycle per add would show up as a
 * length dependence); the TSC is invariant at 2.0 GHz on this part, so the two together
 * give the boost ratio. Accumulators are *named registers*, not an indexed array: an array
 * makes GCC spill every accumulator to the stack and the probe then measures store-forwarding
 * latency instead of the instruction (that mistake cost a factor of 4 the first time this
 * probe was written, so it is called out here).
 *
 * Three clocks are reported for every probe (CLOCK_MONOTONIC_RAW, the scheduler's CPU
 * accounting, and the TSC), because this host's CLOCK_MONOTONIC runs ~8 % fast; every rate
 * below is the *minimum* of `reps` timings, i.e. the least-descheduled pass.
 *
 * vpdpbusd is VNNI's byte dot product: 4 x 4 bytes per 128-bit lane summed into int32. In
 * a ymm that is 32 multiply-accumulates, in a zmm 64.
 *
 *   gcc -O2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mfma peak.c -o peak
 *   taskset -c 4 ./peak 7
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <immintrin.h>
#include <x86intrin.h>

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

typedef struct { double raw, cpu, tsc_ghz; } clocks;

static clocks bench(void (*fn)(long), long iters, int reps)
{
    clocks best = {1e9, 1e9, 0};
    for (int r = 0; r < reps; r++) {
        double r0 = mono_raw(), c0 = cpu_seconds();
        unsigned long long t0 = __rdtsc();
        fn(iters);
        unsigned long long t1 = __rdtsc();
        double c1 = cpu_seconds(), r1 = mono_raw();
        double raw = r1 - r0, cpu = c1 - c0;
        double v = raw < cpu ? raw : cpu;
        double bv = best.raw < best.cpu ? best.raw : best.cpu;
        if (v < bv) {
            best.raw = raw;
            best.cpu = cpu;
            best.tsc_ghz = (double)(t1 - t0) / 1e9 / v;
        }
    }
    return best;
}

/* ---- the clock: chains of dependent 1-cycle adds ------------------------------------- */
static volatile long sink;
#define ADDS8 \
    "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t" \
    "add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\t"
static void chain8(long n)
{
    long a = 1;
    for (long i = 0; i < n / 8; i++)
        __asm__ volatile(ADDS8 : "+r"(a));
    sink = a;
}
static void chain16(long n)
{
    long a = 1;
    for (long i = 0; i < n / 16; i++)
        __asm__ volatile(ADDS8 ADDS8 : "+r"(a));
    sink = a;
}

static void chain_imul(long n)
{
    long a = 3;
    for (long i = 0; i < n; i++)
        __asm__ volatile("imul %1, %0" : "+r"(a) : "r"(5L));
    sink = a;
}

/* ---- vpdpbusd, register operands, named accumulators --------------------------------- */
#define Z _mm512_setzero_si512()
#define Y _mm256_setzero_si256()
/* Inline asm, not the intrinsic: GCC's CSE folds equal accumulators into one and a final
 * multiply (all sixteen ran as one chain with two vmovdqa when the intrinsic was used), so
 * the probe measured a dependency chain instead of the instruction. */
#define D16(acc, a, b) __asm__ volatile("vpdpbusd %[a], %[b], %[c]" \
    : [c] "+v"(acc) : [a] "v"(a), [b] "v"(b))
#define D8(acc, a, b)  __asm__ volatile("vpdpbusd %[a], %[b], %[c]" \
    : [c] "+x"(acc) : [a] "x"(a), [b] "x"(b))

static void dp_zmm_1(long n)
{
    __m512i a = _mm512_set1_epi8(7), b = _mm512_set1_epi8(3), c0 = Z;
    for (long i = 0; i < n; i++) D16(c0, a, b);
    sink = _mm512_reduce_add_epi32(c0);
}
static void dp_zmm_2(long n)
{
    __m512i a = _mm512_set1_epi8(7), b = _mm512_set1_epi8(3), c0 = Z, c1 = Z;
    for (long i = 0; i < n; i++) { D16(c0,a,b); D16(c1,a,b); }
    sink = _mm512_reduce_add_epi32(_mm512_add_epi32(c0, c1));
}
static void dp_zmm_4(long n)
{
    __m512i a = _mm512_set1_epi8(7), b = _mm512_set1_epi8(3), c0 = Z, c1 = Z, c2 = Z, c3 = Z;
    for (long i = 0; i < n; i++) { D16(c0,a,b); D16(c1,a,b); D16(c2,a,b); D16(c3,a,b); }
    sink = _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(c0,c1),
                                                    _mm512_add_epi32(c2,c3)));
}
static void dp_zmm_8(long n)
{
    __m512i a = _mm512_set1_epi8(7), b = _mm512_set1_epi8(3);
    __m512i c0=Z,c1=Z,c2=Z,c3=Z,c4=Z,c5=Z,c6=Z,c7=Z;
    for (long i = 0; i < n; i++) {
        D16(c0,a,b); D16(c1,a,b); D16(c2,a,b); D16(c3,a,b);
        D16(c4,a,b); D16(c5,a,b); D16(c6,a,b); D16(c7,a,b);
    }
    sink = _mm512_reduce_add_epi32(_mm512_add_epi32(
        _mm512_add_epi32(_mm512_add_epi32(c0,c1), _mm512_add_epi32(c2,c3)),
        _mm512_add_epi32(_mm512_add_epi32(c4,c5), _mm512_add_epi32(c6,c7))));
}
static void dp_zmm_16(long n)
{
    __m512i a = _mm512_set1_epi8(7), b = _mm512_set1_epi8(3);
    __m512i c0=Z,c1=Z,c2=Z,c3=Z,c4=Z,c5=Z,c6=Z,c7=Z;
    __m512i d0=Z,d1=Z,d2=Z,d3=Z,d4=Z,d5=Z,d6=Z,d7=Z;
    for (long i = 0; i < n; i++) {
        D16(c0,a,b); D16(c1,a,b); D16(c2,a,b); D16(c3,a,b);
        D16(c4,a,b); D16(c5,a,b); D16(c6,a,b); D16(c7,a,b);
        D16(d0,a,b); D16(d1,a,b); D16(d2,a,b); D16(d3,a,b);
        D16(d4,a,b); D16(d5,a,b); D16(d6,a,b); D16(d7,a,b);
    }
    __m512i s = _mm512_add_epi32(
        _mm512_add_epi32(_mm512_add_epi32(c0,c1), _mm512_add_epi32(c2,c3)),
        _mm512_add_epi32(_mm512_add_epi32(c4,c5), _mm512_add_epi32(c6,c7)));
    __m512i t = _mm512_add_epi32(
        _mm512_add_epi32(_mm512_add_epi32(d0,d1), _mm512_add_epi32(d2,d3)),
        _mm512_add_epi32(_mm512_add_epi32(d4,d5), _mm512_add_epi32(d6,d7)));
    sink = _mm512_reduce_add_epi32(_mm512_add_epi32(s, t));
}

static int ymm_reduce(__m256i v)
{
    __m128i h = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0x4e));
    h = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0xb1));
    return _mm_cvtsi128_si32(h);
}
static void dp_ymm_1(long n)
{
    __m256i a = _mm256_set1_epi8(7), b = _mm256_set1_epi8(3), c0 = Y;
    for (long i = 0; i < n; i++) D8(c0, a, b);
    sink = ymm_reduce(c0);
}
static void dp_ymm_4(long n)
{
    __m256i a = _mm256_set1_epi8(7), b = _mm256_set1_epi8(3), c0 = Y, c1 = Y, c2 = Y, c3 = Y;
    for (long i = 0; i < n; i++) { D8(c0,a,b); D8(c1,a,b); D8(c2,a,b); D8(c3,a,b); }
    sink = ymm_reduce(_mm256_add_epi32(_mm256_add_epi32(c0,c1), _mm256_add_epi32(c2,c3)));
}
static void dp_ymm_16(long n)
{
    __m256i a = _mm256_set1_epi8(7), b = _mm256_set1_epi8(3);
    __m256i c0=Y,c1=Y,c2=Y,c3=Y,c4=Y,c5=Y,c6=Y,c7=Y;
    __m256i d0=Y,d1=Y,d2=Y,d3=Y,d4=Y,d5=Y,d6=Y,d7=Y;
    for (long i = 0; i < n; i++) {
        D8(c0,a,b); D8(c1,a,b); D8(c2,a,b); D8(c3,a,b);
        D8(c4,a,b); D8(c5,a,b); D8(c6,a,b); D8(c7,a,b);
        D8(d0,a,b); D8(d1,a,b); D8(d2,a,b); D8(d3,a,b);
        D8(d4,a,b); D8(d5,a,b); D8(d6,a,b); D8(d7,a,b);
    }
    __m256i s = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_add_epi32(c0,c1), _mm256_add_epi32(c2,c3)),
        _mm256_add_epi32(_mm256_add_epi32(c4,c5), _mm256_add_epi32(c6,c7)));
    __m256i t = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_add_epi32(d0,d1), _mm256_add_epi32(d2,d3)),
        _mm256_add_epi32(_mm256_add_epi32(d4,d5), _mm256_add_epi32(d6,d7)));
    sink = ymm_reduce(_mm256_add_epi32(s, t));
}

/* ---- vpdpbusd, zmm, second operand from an L1 buffer -------------------------------- */
static uint8_t *buf;
static void dp_zmm_mem_4(long n)
{
    __m512i a = _mm512_set1_epi8(7), c0 = Z, c1 = Z, c2 = Z, c3 = Z;
    const uint8_t *p0 = buf, *p1 = buf + 64, *p2 = buf + 128, *p3 = buf + 192;
    for (long i = 0; i < n; i++) {
        __asm__ volatile("vpdpbusd (%[m]), %[b], %[c]" : [c] "+v"(c0) : [b] "v"(a), [m] "r"(p0));
        __asm__ volatile("vpdpbusd (%[m]), %[b], %[c]" : [c] "+v"(c1) : [b] "v"(a), [m] "r"(p1));
        __asm__ volatile("vpdpbusd (%[m]), %[b], %[c]" : [c] "+v"(c2) : [b] "v"(a), [m] "r"(p2));
        __asm__ volatile("vpdpbusd (%[m]), %[b], %[c]" : [c] "+v"(c3) : [b] "v"(a), [m] "r"(p3));
    }
    sink = _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(c0,c1),
                                                    _mm512_add_epi32(c2,c3)));
}

/* ---- f32 FMA, for the clock/width cross-check ---------------------------------------- */
static void fma_zmm_8(long n)
{
    __m512 a = _mm512_set1_ps(1.0000001f);
    const __m512 F = _mm512_setzero_ps();
    __m512 c0=F,c1=F,c2=F,c3=F,c4=F,c5=F,c6=F,c7=F;
    for (long i = 0; i < n; i++) {
        c0=_mm512_fmadd_ps(c0,a,a); c1=_mm512_fmadd_ps(c1,a,a);
        c2=_mm512_fmadd_ps(c2,a,a); c3=_mm512_fmadd_ps(c3,a,a);
        c4=_mm512_fmadd_ps(c4,a,a); c5=_mm512_fmadd_ps(c5,a,a);
        c6=_mm512_fmadd_ps(c6,a,a); c7=_mm512_fmadd_ps(c7,a,a);
    }
    __m512 s = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(c0,c1), _mm512_add_ps(c2,c3)),
                             _mm512_add_ps(_mm512_add_ps(c4,c5), _mm512_add_ps(c6,c7)));
    sink = (long)_mm512_reduce_add_ps(s);
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 7;
    long n = 50000000;
    buf = aligned_alloc(64, 1 << 16);
    memset(buf, 1, 1 << 16);

    clocks c8 = bench(chain8, 8L * n, reps), c16 = bench(chain16, 16L * n, reps);
    double t8 = c8.raw < c8.cpu ? c8.raw : c8.cpu, t16 = c16.raw < c16.cpu ? c16.raw : c16.cpu;
    double g8 = (double)(8L * n) / t8, g16 = (double)(16L * n) / t16;
    double ghz = (g8 > g16 ? g8 : g16) / 1e9;   /* GHz */
    clocks ci = bench(chain_imul, n, reps);
    double ti = ci.raw < ci.cpu ? ci.raw : ci.cpu;
    double gimul = (double)n / ti;
    printf("clock: dependent add chains %.3f G/s (unroll 8) and %.3f G/s (unroll 16), "
           "imul chain %.3f G/s; TSC %.3f GHz\n"
           "       => core %.2f GHz if add is 1 cycle (imul then reads %.2f cycles); "
           "TSC %.3f GHz, core/tsc %.2f\n\n",
           g8 / 1e9, g16 / 1e9, gimul / 1e9, c8.tsc_ghz,
           ghz, 3.0 * gimul / 1e9 / ghz * 3.0, c8.tsc_ghz, ghz / c8.tsc_ghz);

    printf("%-13s %8s %9s %10s %6s %s\n",
           "probe", "instr/s G", "GMAC/s", "GMAC/cyc", "IPC", "MACs/instr");
    struct { const char *name; void (*fn)(long); double macs, per_iter; } probes[] = {
        {"dp_zmm_1",    dp_zmm_1,     64,  1},
        {"dp_zmm_2",    dp_zmm_2,     64,  2},
        {"dp_zmm_4",    dp_zmm_4,     64,  4},
        {"dp_zmm_8",    dp_zmm_8,     64,  8},
        {"dp_zmm_16",   dp_zmm_16,    64, 16},
        {"dp_ymm_1",    dp_ymm_1,     32,  1},
        {"dp_ymm_4",    dp_ymm_4,     32,  4},
        {"dp_ymm_16",   dp_ymm_16,    32, 16},
        {"dp_zmm_mem4", dp_zmm_mem_4, 64,  4},
        {"fma_zmm_8",   fma_zmm_8,    16,  8},
    };
    for (unsigned i = 0; i < sizeof probes / sizeof probes[0]; i++) {
        clocks c = bench(probes[i].fn, n, reps);
        double t = c.raw < c.cpu ? c.raw : c.cpu;
        double ips = probes[i].per_iter * (double)n / t;      /* instructions per second */
        double gmac = probes[i].macs * ips / 1e9;
        printf("%-13s %8.2f %9.1f %10.2f %6.2f %.0f\n",
               probes[i].name, ips / 1e9, gmac, gmac / ghz, ips / 1e9 / ghz, probes[i].macs);
    }
    free(buf);
    return 0;
}
