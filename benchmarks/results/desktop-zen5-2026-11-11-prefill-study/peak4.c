/* Clean per-instruction throughput: 8 independent accumulators, constant but distinct
 * operands (nothing to hoist, no chain beyond the accumulator itself). */
#include <immintrin.h>
#include <stdio.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
#define M 50000000L
int main(void){
    /* ymm vpdpbusd: 32 byte-MAC per instruction */
    __m256i c0[8] = {0}, *a = c0;
    __m256i u0=_mm256_set1_epi8(1),u1=_mm256_set1_epi8(3),u2=_mm256_set1_epi8(5),u3=_mm256_set1_epi8(7),
            u4=_mm256_set1_epi8(9),u5=_mm256_set1_epi8(11),u6=_mm256_set1_epi8(13),u7=_mm256_set1_epi8(15);
    double t0=now();
    for(long i=0;i<M;i++){
        a[0]=_mm256_dpbusd_epi32(a[0],u0,u1); a[1]=_mm256_dpbusd_epi32(a[1],u1,u2);
        a[2]=_mm256_dpbusd_epi32(a[2],u2,u3); a[3]=_mm256_dpbusd_epi32(a[3],u3,u4);
        a[4]=_mm256_dpbusd_epi32(a[4],u4,u5); a[5]=_mm256_dpbusd_epi32(a[5],u5,u6);
        a[6]=_mm256_dpbusd_epi32(a[6],u6,u7); a[7]=_mm256_dpbusd_epi32(a[7],u7,u0);
    }
    double t1=now();
    printf("vpdpbusd ymm   %7.1f GMAC/s   %5.2f G instr/s\n", 8.0*M*32.0/(t1-t0)/1e9, 8.0*M/(t1-t0)/1e9);
    /* zmm again, same shape */
    __m512i z[8]; for(int k=0;k<8;k++) z[k]=_mm512_setzero_si512();
    __m512i v0=_mm512_set1_epi8(1),v1=_mm512_set1_epi8(3),v2=_mm512_set1_epi8(5),v3=_mm512_set1_epi8(7),
            v4=_mm512_set1_epi8(9),v5=_mm512_set1_epi8(11),v6=_mm512_set1_epi8(13),v7=_mm512_set1_epi8(15);
    t0=now();
    for(long i=0;i<M;i++){
        z[0]=_mm512_dpbusd_epi32(z[0],v0,v1); z[1]=_mm512_dpbusd_epi32(z[1],v1,v2);
        z[2]=_mm512_dpbusd_epi32(z[2],v2,v3); z[3]=_mm512_dpbusd_epi32(z[3],v3,v4);
        z[4]=_mm512_dpbusd_epi32(z[4],v4,v5); z[5]=_mm512_dpbusd_epi32(z[5],v5,v6);
        z[6]=_mm512_dpbusd_epi32(z[6],v6,v7); z[7]=_mm512_dpbusd_epi32(z[7],v7,v0);
    }
    t1=now();
    printf("vpdpbusd zmm   %7.1f GMAC/s   %5.2f G instr/s\n", 8.0*M*64.0/(t1-t0)/1e9, 8.0*M/(t1-t0)/1e9);
    printf("checksums %d %d\n", _mm256_extract_epi32(a[0],0), _mm512_reduce_add_epi32(z[3]));
    return 0;
}
