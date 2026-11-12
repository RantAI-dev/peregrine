The old AVX2 weight loop (gemm_avx2, origin/main at bf1193e), one k step, disassembled with
`objdump -d --no-show-raw-insn` -- two panel loads, six broadcasts, twelve FMAs, and per step
the prefetch guard (test/je) plus the format match (cmpb/je for F32, cmpl/jne for F16).

   5fee0:	vmovdqu (%r9),%ymm14
   5fee5:	vmovdqu 0x20(%r9),%ymm15
   5feeb:	vbroadcastss (%rax,%rbp,4),%ymm0
   5fef1:	vfmadd231ps %ymm14,%ymm0,%ymm13
   5fef6:	vfmadd231ps %ymm0,%ymm15,%ymm12
   5fefb:	vbroadcastss (%rsi,%rbp,4),%ymm0
   5ff01:	vfmadd231ps %ymm14,%ymm0,%ymm11
   5ff06:	vfmadd231ps %ymm0,%ymm15,%ymm10
   5ff0b:	vbroadcastss (%r10,%rbp,4),%ymm0
   5ff11:	vfmadd231ps %ymm14,%ymm0,%ymm9
   5ff16:	vfmadd231ps %ymm0,%ymm15,%ymm8
   5ff1b:	vbroadcastss (%r8,%rbp,4),%ymm0
   5ff21:	vfmadd231ps %ymm14,%ymm0,%ymm7
   5ff26:	vfmadd231ps %ymm0,%ymm15,%ymm6
   5ff2b:	vbroadcastss (%r15,%rbp,4),%ymm0
   5ff31:	vfmadd231ps %ymm14,%ymm0,%ymm5
   5ff36:	vfmadd231ps %ymm0,%ymm15,%ymm4
   5ff3b:	vbroadcastss (%r14,%rbp,4),%ymm0
   5ff41:	inc    %rbp
   5ff44:	vfmadd231ps %ymm14,%ymm0,%ymm2
   5ff49:	vfmadd231ps %ymm15,%ymm0,%ymm3
   5ff4e:	add    %r13,%r9
   5ff51:	cmp    %rbp,%rcx
   5ff54:	je     5ffb0 <_RNvNtNtCsi5F2HisjK6H_17peregrine_kernels5fpack3x869gemm_avx2+0x460>
   5ff56:	test   %r12,%r12
   5ff59:	je     5ff60 <_RNvNtNtCsi5F2HisjK6H_17peregrine_kernels5fpack3x869gemm_avx2+0x410>
   5ff5b:	prefetcht0 (%r9,%r12,1)
   5ff60:	cmpb   $0x0,0xc(%rsp)
