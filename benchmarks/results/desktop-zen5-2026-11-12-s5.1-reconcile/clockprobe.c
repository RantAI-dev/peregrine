/* The host's clocks, side by side, for one pinned busy loop.
 *
 * A one-thread loop that is never descheduled must show the same elapsed time on every
 * clock. This prints four of them so a skewed clock is visible instead of inferred:
 * CLOCK_MONOTONIC and CLOCK_MONOTONIC_RAW (which both engines time with), the scheduler's
 * own CPU accounting (getrusage), and the TSC, plus the TSC's implied frequency.
 *
 *   gcc -O2 clockprobe.c -o clockprobe && taskset -c 4 ./clockprobe 3
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>
#include <x86intrin.h>

static double mono(clockid_t c)
{
    struct timespec t;
    clock_gettime(c, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static double cpu_seconds(void)
{
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return (double)(r.ru_utime.tv_sec + r.ru_stime.tv_sec)
         + 1e-6 * (double)(r.ru_utime.tv_usec + r.ru_stime.tv_usec);
}

int main(int argc, char **argv)
{
    double want = argc > 1 ? atof(argv[1]) : 3.0;
    volatile unsigned long x = 0;
    for (unsigned long i = 0; i < 20000000UL; i++) x += i;   /* warm the clock, not the caches */

    double m0 = mono(CLOCK_MONOTONIC), r0 = mono(CLOCK_MONOTONIC_RAW), c0 = cpu_seconds();
    unsigned long long t0 = __rdtsc();
    while (mono(CLOCK_MONOTONIC) - m0 < want)
        for (unsigned long i = 0; i < 1000000UL; i++) x += i;
    unsigned long long t1 = __rdtsc();
    double c1 = cpu_seconds(), r1 = mono(CLOCK_MONOTONIC_RAW), m1 = mono(CLOCK_MONOTONIC);

    double w = m1 - m0, raw = r1 - r0, cpu = c1 - c0, ticks = (double)(t1 - t0);
    printf("MONOTONIC %.3f  RAW %.3f  CPU %.3f  TSC/Gticks %.3f  "
           "tsc_ghz_if_cpu_accurate %.3f  cpu_over_wall %.3f\n",
           w, raw, cpu, ticks / 1e9, ticks / 1e9 / cpu, cpu / w);
    (void)x;
    return 0;
}
