/* vpdpbusd ymm with 16 scalar accumulators (seeded from a volatile, so nothing folds). */
#include <immintrin.h>
#include <stdio.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
#define M 20000000L
int main(void){
    volatile int s = 1;
    __m256i z0=_mm256_setzero_si256(),z1=z0,z2=z0,z3=z0,z4=z0,z5=z0,z6=z0,z7=z0,
            z8=z0,z9=z0,za=z0,zb=z0,zc=z0,zd=z0,ze=z0,zf=z0;
    __m256i a=_mm256_set1_epi8(s),b=_mm256_set1_epi8(s+2),c=_mm256_set1_epi8(s+4),d=_mm256_set1_epi8(s+6),
            e=_mm256_set1_epi8(s+8),f=_mm256_set1_epi8(s+10),g=_mm256_set1_epi8(s+12),h=_mm256_set1_epi8(s+14);
    double t0=now();
    for (long i=0;i<M;i++){
        z0=_mm256_dpbusd_epi32(z0,a,b); z1=_mm256_dpbusd_epi32(z1,b,c);
        z2=_mm256_dpbusd_epi32(z2,c,d); z3=_mm256_dpbusd_epi32(z3,d,e);
        z4=_mm256_dpbusd_epi32(z4,e,f); z5=_mm256_dpbusd_epi32(z5,f,g);
        z6=_mm256_dpbusd_epi32(z6,g,h); z7=_mm256_dpbusd_epi32(z7,h,a);
        z8=_mm256_dpbusd_epi32(z8,a,c); z9=_mm256_dpbusd_epi32(z9,b,d);
        za=_mm256_dpbusd_epi32(za,c,e); zb=_mm256_dpbusd_epi32(zb,d,f);
        zc=_mm256_dpbusd_epi32(zc,e,g); zd=_mm256_dpbusd_epi32(zd,f,h);
        ze=_mm256_dpbusd_epi32(ze,g,a); zf=_mm256_dpbusd_epi32(zf,h,b);
    }
    double t1=now();
    printf("dpbusd ymm 16 %7.1f GMAC/s  %5.2f G instr/s\n", 16.0*M*32.0/(t1-t0)/1e9, 16.0*M/(t1-t0)/1e9);
    printf("checksum %d\n", _mm256_extract_epi32(z0,0));
    return 0;
}
