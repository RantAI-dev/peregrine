/* Which instruction sets the fork's library was compiled with, read back from the library
 * itself instead of from its banner. The S5.1 review asked whether the probe was linked
 * against the same build the server ran; this prints the library's answer.
 *
 *   gcc -O2 -I $FORK/ggml/include isa.c -o isa -L $FORK/build/ggml/src -lggml \
 *       -Wl,-rpath,$FORK/build/ggml/src && ./isa
 */
#include "ggml.h"
#include <stdio.h>

int main(void)
{
    printf("avx=%d avx2=%d fma=%d f16c=%d avx512=%d avx512_vbmi=%d avx512_vnni=%d "
           "avx512_bf16=%d matmul_int8=%d\n",
           ggml_cpu_has_avx(), ggml_cpu_has_avx2(), ggml_cpu_has_fma(), ggml_cpu_has_f16c(),
           ggml_cpu_has_avx512(), ggml_cpu_has_avx512_vbmi(), ggml_cpu_has_avx512_vnni(),
           ggml_cpu_has_avx512_bf16(), ggml_cpu_has_matmul_int8());
    return 0;
}
