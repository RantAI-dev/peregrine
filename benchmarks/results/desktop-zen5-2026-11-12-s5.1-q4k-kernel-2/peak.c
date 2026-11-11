// The byte-dot roof this core actually reaches: 16 named accumulators, no arrays, one
// asm barrier per iteration so nothing can be hoisted, 512-bit and 256-bit forms.
#include <immintrin.h>
#include <stdio.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
static unsigned char buf[4096];
#define REP 2000000
static __m512i z;
static __m256i y;
#define ZACC c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15
int main(void){
  for (unsigned i=0;i<sizeof buf;i++) buf[i]=(unsigned char)(i*37+11);
  __m512i w = _mm512_loadu_si512(buf);
  __m256i v = _mm256_loadu_si256((const __m256i*)buf);
  __m512i a = _mm512_set1_epi64(0x0102030405060708LL);
  __m256i b = _mm256_set1_epi64x(0x0102030405060708LL);
  double t0,t1;

  { __m512i ZACC;
    c0=c1=c2=c3=c4=c5=c6=c7=_mm512_setzero_si512();
    c8=c9=c10=c11=c12=c13=c14=c15=_mm512_setzero_si512();
    t0=now();
    for(long r=0;r<REP;r++){
      c0=_mm512_dpbusd_epi32(c0,w,a); c1=_mm512_dpbusd_epi32(c1,w,a);
      c2=_mm512_dpbusd_epi32(c2,w,a); c3=_mm512_dpbusd_epi32(c3,w,a);
      c4=_mm512_dpbusd_epi32(c4,w,a); c5=_mm512_dpbusd_epi32(c5,w,a);
      c6=_mm512_dpbusd_epi32(c6,w,a); c7=_mm512_dpbusd_epi32(c7,w,a);
      c8=_mm512_dpbusd_epi32(c8,w,a); c9=_mm512_dpbusd_epi32(c9,w,a);
      c10=_mm512_dpbusd_epi32(c10,w,a); c11=_mm512_dpbusd_epi32(c11,w,a);
      c12=_mm512_dpbusd_epi32(c12,w,a); c13=_mm512_dpbusd_epi32(c13,w,a);
      c14=_mm512_dpbusd_epi32(c14,w,a); c15=_mm512_dpbusd_epi32(c15,w,a);
      __asm__ __volatile__("" : "+v"(c0),"+v"(c1),"+v"(c2),"+v"(c3),"+v"(c4),"+v"(c5),"+v"(c6),"+v"(c7));
      __asm__ __volatile__("" : "+v"(c8),"+v"(c9),"+v"(c10),"+v"(c11),"+v"(c12),"+v"(c13),"+v"(c14),"+v"(c15));
    }
    t1=now(); z=_mm512_add_epi32(c0,c15);
    printf("dpbusd zmm: %.2f G instr/s  %.1f GMAC/s\n", REP*16/(t1-t0)/1e9, REP*16*64.0/(t1-t0)/1e9); }

  { __m256i d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15;
    d0=d1=d2=d3=d4=d5=d6=d7=_mm256_setzero_si256();
    d8=d9=d10=d11=d12=d13=d14=d15=_mm256_setzero_si256();
    t0=now();
    for(long r=0;r<REP;r++){
      d0=_mm256_dpbusd_epi32(d0,v,b); d1=_mm256_dpbusd_epi32(d1,v,b);
      d2=_mm256_dpbusd_epi32(d2,v,b); d3=_mm256_dpbusd_epi32(d3,v,b);
      d4=_mm256_dpbusd_epi32(d4,v,b); d5=_mm256_dpbusd_epi32(d5,v,b);
      d6=_mm256_dpbusd_epi32(d6,v,b); d7=_mm256_dpbusd_epi32(d7,v,b);
      d8=_mm256_dpbusd_epi32(d8,v,b); d9=_mm256_dpbusd_epi32(d9,v,b);
      d10=_mm256_dpbusd_epi32(d10,v,b); d11=_mm256_dpbusd_epi32(d11,v,b);
      d12=_mm256_dpbusd_epi32(d12,v,b); d13=_mm256_dpbusd_epi32(d13,v,b);
      d14=_mm256_dpbusd_epi32(d14,v,b); d15=_mm256_dpbusd_epi32(d15,v,b);
      __asm__ __volatile__("" : "+v"(d0),"+v"(d1),"+v"(d2),"+v"(d3),"+v"(d4),"+v"(d5),"+v"(d6),"+v"(d7));
      __asm__ __volatile__("" : "+v"(d8),"+v"(d9),"+v"(d10),"+v"(d11),"+v"(d12),"+v"(d13),"+v"(d14),"+v"(d15));
    }
    t1=now(); y=_mm256_add_epi32(d0,d15);
    printf("dpbusd ymm: %.2f G instr/s  %.1f GMAC/s\n", REP*16/(t1-t0)/1e9, REP*16*32.0/(t1-t0)/1e9); }

  { __m512i e0,e1,e2,e3,e4,e5,e6,e7;
    e0=e1=e2=e3=_mm512_setzero_si512(); e4=e5=e6=e7=_mm512_setzero_si512();
    __m512 f=_mm512_set1_ps(1.0000001f), g=_mm512_set1_ps(1e-9f);
    t0=now();
    for(long r=0;r<REP;r++){
      e0=_mm512_castps_si512(_mm512_fmadd_ps(_mm512_castsi512_ps(e0),f,g));
      e1=_mm512_castps_si512(_mm512_fmadd_ps(_mm512_castsi512_ps(e1),f,g));
      e2=_mm512_castps_si512(_mm512_fmadd_ps(_mm512_castsi512_ps(e2),f,g));
      e3=_mm512_castps_si512(_mm512_fmadd_ps(_mm512_castsi512_ps(e3),f,g));
      e4=_mm512_castps_si512(_mm512_fmadd_ps(_mm512_castsi512_ps(e4),f,g));
      e5=_mm512_castps_si512(_mm512_fmadd_ps(_mm512_castsi512_ps(e5),f,g));
      e6=_mm512_castps_si512(_mm512_fmadd_ps(_mm512_castsi512_ps(e6),f,g));
      e7=_mm512_castps_si512(_mm512_fmadd_ps(_mm512_castsi512_ps(e7),f,g));
      __asm__ __volatile__("" : "+v"(e0),"+v"(e1),"+v"(e2),"+v"(e3),"+v"(e4),"+v"(e5),"+v"(e6),"+v"(e7));
    }
    t1=now(); z=_mm512_add_epi32(e0,e7);
    printf("vfmadd zmm:  %.2f G instr/s\n", REP*8/(t1-t0)/1e9); }
  __asm__ __volatile__("" :: "v"(z), "v"(y));
  return 0;
}
