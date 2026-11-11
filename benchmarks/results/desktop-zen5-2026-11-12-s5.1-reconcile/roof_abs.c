/* The byte-dot roof against a clock outside the VM.
 *
 * The guest's own clocks disagree with each other and the disagreement changes between
 * runs, so this probe does not believe any of them: it runs a fixed number of
 * `vpdpbusd` instructions (16 named accumulators, one instruction per accumulator per
 * iteration, operands in registers) and the *shell* brackets the run with two remote
 * HTTP Date headers. MACs / host seconds is then a rate in real seconds, whatever the
 * guest thinks the time is. It prints its own three clocks beside it, for comparison.
 *
 *   gcc -O2 -mavx512f -mavx512bw -mavx512vl -mavx512vnni roof_abs.c -o roof_abs
 *   taskset -c 4 ./roof_abs 2000000000
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>
#include <immintrin.h>
#include <x86intrin.h>

static double mono(clockid_t c){struct timespec t;clock_gettime(c,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static double cpu(void){struct rusage r;getrusage(RUSAGE_SELF,&r);
 return r.ru_utime.tv_sec+1e-6*r.ru_utime.tv_usec+r.ru_stime.tv_sec+1e-6*r.ru_stime.tv_usec;}
static volatile long sink;

int main(int argc, char **argv)
{
    long iters = argc > 1 ? atol(argv[1]) : 2000000000L;
    __m512i a = _mm512_set1_epi8(7), b = _mm512_set1_epi8(3);
    __m512i c0=_mm512_setzero_si512(), c1=c0, c2=c0, c3=c0, c4=c0, c5=c0, c6=c0, c7=c0;
    __m512i d0=c0, d1=c0, d2=c0, d3=c0, d4=c0, d5=c0, d6=c0, d7=c0;
#define D(x) __asm__ volatile("vpdpbusd %[a], %[b], %[c]" : [c] "+v"(x) : [a] "v"(a), [b] "v"(b))
    double m0 = mono(CLOCK_MONOTONIC), r0 = mono(CLOCK_MONOTONIC_RAW), k0 = cpu();
    unsigned long long t0 = __rdtsc();
    for (long i = 0; i < iters; i++) {
        D(c0); D(c1); D(c2); D(c3); D(c4); D(c5); D(c6); D(c7);
        D(d0); D(d1); D(d2); D(d3); D(d4); D(d5); D(d6); D(d7);
    }
    unsigned long long t1 = __rdtsc();
    double k1 = cpu(), r1 = mono(CLOCK_MONOTONIC_RAW), m1 = mono(CLOCK_MONOTONIC);
    double macs = (double)iters * 16.0 * 64.0;
    printf("instr %ld  MACs %.3e\n", iters * 16, macs);
    printf("mono %.3f  raw %.3f  cpu %.3f  tsc %.3f  (Gticks %.3f)\n",
           m1 - m0, r1 - r0, k1 - k0, (double)(t1 - t0) / 1e9, (double)(t1 - t0) / 1e9);
    printf("rate by own clocks: mono %.1f  raw %.1f  cpu %.1f GMAC/s\n",
           macs / (m1 - m0) / 1e9, macs / (r1 - r0) / 1e9, macs / (k1 - k0) / 1e9);
    (void)sink;
    return 0;
}
