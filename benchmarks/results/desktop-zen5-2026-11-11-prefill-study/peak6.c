/* Instruction roofs, 16 scalar accumulators (registers, no arrays) and constant
 * operands, so the only limit is the instruction's own throughput. */
#include <immintrin.h>
#include <stdio.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
#define M 20000000L
int main(void){
    __m512i z0=_mm512_setzero_si512(),z1=z0,z2=z0,z3=z0,z4=z0,z5=z0,z6=z0,z7=z0,
            z8=z0,z9=z0,za=z0,zb=z0,zc=z0,zd=z0,ze=z0,zf=z0;
    __m512i a=_mm512_set1_epi8(1),b=_mm512_set1_epi8(3),c=_mm512_set1_epi8(5),d=_mm512_set1_epi8(7),
            e=_mm512_set1_epi8(9),f=_mm512_set1_epi8(11),g=_mm512_set1_epi8(13),h=_mm512_set1_epi8(15);
    double t0=now();
    for (long i=0;i<M;i++){
        z0=_mm512_dpbusd_epi32(z0,a,b); z1=_mm512_dpbusd_epi32(z1,b,c);
        z2=_mm512_dpbusd_epi32(z2,c,d); z3=_mm512_dpbusd_epi32(z3,d,e);
        z4=_mm512_dpbusd_epi32(z4,e,f); z5=_mm512_dpbusd_epi32(z5,f,g);
        z6=_mm512_dpbusd_epi32(z6,g,h); z7=_mm512_dpbusd_epi32(z7,h,a);
        z8=_mm512_dpbusd_epi32(z8,a,c); z9=_mm512_dpbusd_epi32(z9,b,d);
        za=_mm512_dpbusd_epi32(za,c,e); zb=_mm512_dpbusd_epi32(zb,d,f);
        zc=_mm512_dpbusd_epi32(zc,e,g); zd=_mm512_dpbusd_epi32(zd,f,h);
        ze=_mm512_dpbusd_epi32(ze,g,a); zf=_mm512_dpbusd_epi32(zf,h,b);
    }
    double t1=now();
    printf("dpbusd zmm 16 %7.1f GMAC/s  %5.2f G instr/s\n", 16.0*M*64.0/(t1-t0)/1e9, 16.0*M/(t1-t0)/1e9);
    __m512 p0=_mm512_set1_ps(1.0f),p1=p0,p2=p0,p3=p0,p4=p0,p5=p0,p6=p0,p7=p0,
          p8=p0,p9=p0,pa=p0,pb=p0,pc=p0,pd=p0,pe=p0,pf=p0;
    __m512 cc=_mm512_set1_ps(1e-9f), dd=_mm512_set1_ps(1e-7f);
    t0=now();
    for (long i=0;i<M;i++){
        p0=_mm512_fmadd_ps(p0,cc,dd); p1=_mm512_fmadd_ps(p1,cc,dd);
        p2=_mm512_fmadd_ps(p2,cc,dd); p3=_mm512_fmadd_ps(p3,cc,dd);
        p4=_mm512_fmadd_ps(p4,cc,dd); p5=_mm512_fmadd_ps(p5,cc,dd);
        p6=_mm512_fmadd_ps(p6,cc,dd); p7=_mm512_fmadd_ps(p7,cc,dd);
        p8=_mm512_fmadd_ps(p8,cc,dd); p9=_mm512_fmadd_ps(p9,cc,dd);
        pa=_mm512_fmadd_ps(pa,cc,dd); pb=_mm512_fmadd_ps(pb,cc,dd);
        pc=_mm512_fmadd_ps(pc,cc,dd); pd=_mm512_fmadd_ps(pd,cc,dd);
        pe=_mm512_fmadd_ps(pe,cc,dd); pf=_mm512_fmadd_ps(pf,cc,dd);
    }
    t1=now();
    printf("fma zmm 16     %7.2f G instr/s\n", 16.0*M/(t1-t0)/1e9);
    printf("checksum %d %f\n", _mm512_reduce_add_epi32(z0), _mm512_reduce_add_ps(p0));
    return 0;
}
