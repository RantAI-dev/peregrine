/* The core clock, from a chain of dependent 1-cycle `add`s (asm, so it cannot be folded). */
#include <stdio.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(void){
    long n = 37500000;   /* 8 adds per iteration */
    long a = 0;
    double t0 = now();
    __asm__ volatile(
        "1:\n\t"
        "add $1, %0\n\t add $1, %0\n\t add $1, %0\n\t add $1, %0\n\t"
        "add $1, %0\n\t add $1, %0\n\t add $1, %0\n\t add $1, %0\n\t"
        "dec %1\n\t"
        "jnz 1b\n\t"
        : "+r"(a), "+r"(n)
        :
        : "cc");
    double t1 = now();
    printf("%.3f GHz  (checksum %ld)\n", (double)37500000*8.0/(t1-t0)/1e9, a);
    return 0;
}
