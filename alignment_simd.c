#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "alignment_simd.h"

/*
 * #define DEBUG_SIMD
 */

#ifdef __AVX512BW__

#include <immintrin.h>

#define EPU8_TYPE(ins)        ins##_epu8
#define EPI8_TYPE(ins)        ins##_epi8
#define EPI8_MASK_TYPE(ins)   ins##_epi8_mask
#define EPI16_TYPE(ins)       ins##_epi16
#define EPI16_MASK_TYPE(ins)  ins##_epi16_mask
#define EPI32_TYPE(ins)       ins##_epi32
#define EPI32_MASK_TYPE(ins)  ins##_epi32_mask
#define EPI64_TYPE(ins)       ins##_epi64
#define EPI64_MASK_TYPE(ins)  ins##_epi64_mask
#define EPI128_TYPE(ins)      ins##_epi128
#define SI512_TYPE(ins)       ins##_si512
#define VTYPE512                   __m512i

/* data conversion union */
typedef union __attribute__((packed, aligned (64))) {
  VTYPE512 v;
  /* __uint128_t u128[sizeof(VTYPE512)/sizeof(__uint128_t)];*/
  /* [FIXME] not availaible on some compilers (e.g. gcc < 4.4 ; clang ?? ; icc ?? ; ) */
  uint64_t u64[sizeof(VTYPE512)/sizeof(uint64_t)];
  uint32_t u32[sizeof(VTYPE512)/sizeof(uint32_t)];
  uint16_t u16[sizeof(VTYPE512)/sizeof(uint16_t)];
} vector512_t;

#endif


#ifdef __AVX2__

#include <immintrin.h>

#define EPU8_TYPE(ins)   ins##_epu8
#define EPI8_TYPE(ins)   ins##_epi8
#define EPI16_TYPE(ins)  ins##_epi16
#define EPI32_TYPE(ins)  ins##_epi32
#define EPI64_TYPE(ins)  ins##_epi64
#define SI256_TYPE(ins)  ins##_si256
#define VTYPE256            __m256i

/* data conversion union */
typedef union __attribute__((packed, aligned (32))) {
  VTYPE256 v;
  /* __uint128_t u128[sizeof(VTYPE256)/sizeof(__uint128_t)];*/
  /* [FIXME] not availaible on some compilers (e.g. gcc < 4.4 ; clang ?? ; icc ?? ; ) */
  uint64_t u64[sizeof(VTYPE256)/sizeof(uint64_t)];
  uint32_t u32[sizeof(VTYPE256)/sizeof(uint32_t)];
  uint16_t u16[sizeof(VTYPE256)/sizeof(uint16_t)];
} vector256_t;

#endif


#ifdef __SSE2__

#include <emmintrin.h>

#define EPU8_TYPE(ins)   ins##_epu8
#define EPI8_TYPE(ins)   ins##_epi8
#define EPI16_TYPE(ins)  ins##_epi16
#define EPI32_TYPE(ins)  ins##_epi32
#define EPI64_TYPE(ins)  ins##_epi64
#define EPI64X_TYPE(ins) ins##_epi64x
#define SI128_TYPE(ins)  ins##_si128
#define VTYPE128            __m128i

/* data conversion union */
typedef union __attribute__((packed, aligned (16))) {
  VTYPE128 v;
  /* __uint128_t u128[sizeof(VTYPE128)/sizeof(__uint128_t)];*/
  /* [FIXME] not availaible on some compilers (e.g. gcc < 4.4 ; clang ?? ; icc ?? ; ) */
  uint64_t u64[sizeof(VTYPE128)/sizeof(uint64_t)];
  uint32_t u32[sizeof(VTYPE128)/sizeof(uint32_t)];
  uint16_t u16[sizeof(VTYPE128)/sizeof(uint16_t)];
} vector128_t;

#endif


#ifdef __SSE__

#include <xmmintrin.h>

#define PU8_TYPE(ins)  ins##_pu8
#define PI8_TYPE(ins)  ins##_pi8
#define PI16_TYPE(ins) ins##_pi16
#define PI32_TYPE(ins) ins##_pi32
#define SI64_TYPE(ins) ins##_si64
#define VTYPE64           __m64

/* data conversion union */
typedef union __attribute__((packed, aligned (16))) {
  VTYPE64 v;
  uint64_t u64[sizeof(VTYPE64)/sizeof(uint64_t)];
  uint32_t u32[sizeof(VTYPE64)/sizeof(uint32_t)];
  uint16_t u16[sizeof(VTYPE64)/sizeof(uint16_t)];
} vector64_t;

#endif


/*
 * check if the cpu has this instruction set with cpuid
 */


int alignment_avx512bw__compatible_proc() {
#if defined(__x86_64__)
  uint64_t _ax = 0, _bx = 0, _cx = 0, _dx = 0;
  asm volatile (
                "pushq %%rbx" "\n\t"
                "cpuid" "\n\t"
                "movq %%rbx,%1" "\n\t"
                "popq %%rbx" "\n\t"
                : "=a" (_ax), "=r" (_bx), "=c" (_cx), "=d" (_dx)
                : "a" (7), "c" (0)
                );
#ifdef DEBUG_SIMD
  fprintf(stderr,"[compatible avx512bw ? %s]\n", _bx & 1<<30 ? "yes":"no");
#endif
  return (_bx & 1<<30) != 0;
#else
#ifdef DEBUG_SIMD
  fprintf(stderr,"[compatible avx512bw ? no (32bits compiled)]\n");
#endif
  return 0;
#endif
}


int alignment_avx2__compatible_proc() {
#if defined(__x86_64__)
  uint64_t _ax = 0, _bx = 0, _cx = 0, _dx = 0;
  asm volatile (
                "pushq %%rbx" "\n\t"
                "cpuid" "\n\t"
                "movq %%rbx,%1" "\n\t"
                "popq %%rbx" "\n\t"
                : "=a" (_ax), "=r" (_bx), "=c" (_cx), "=d" (_dx)
                : "a" (7), "c" (0)
                );
#ifdef DEBUG_SIMD
  fprintf(stderr,"[compatible avx2 ? %s]\n", _bx & 1<<5 ? "yes":"no");
#endif
  return (_bx & 1<<5) != 0;
#else
#ifdef DEBUG_SIMD
  fprintf(stderr,"[compatible avx2 ? no (32bits compiled)]\n");
#endif
  return 0;
#endif
}


int alignment_sse2__compatible_proc() {
#if defined(__x86_64__)
  uint64_t _ax = 0, _bx = 0, _cx = 0, _dx = 0;
  asm volatile (
                "pushq %%rbx" "\n\t"
                "cpuid" "\n\t"
                "movq %%rbx,%1" "\n\t"
                "popq %%rbx" "\n\t"
                : "=a" (_ax), "=r" (_bx), "=c" (_cx), "=d" (_dx)
                : "a" (1)
                );
#else
  uint32_t _ax = 0, _bx = 0, _cx = 0, _dx = 0;
  asm volatile (
                "pushl %%ebx" "\n\t"
                "cpuid" "\n\t"
                "movl %%ebx,%1" "\n\t"
                "popl %%ebx" "\n\t"
                : "=a" (_ax), "=r" (_bx), "=c" (_cx), "=d" (_dx)
                : "a" (1)
                );
#endif
#ifdef DEBUG_SIMD
  fprintf(stderr,"[compatible sse2 ? %s]\n", _dx & 1<<26 ? "yes":"no");
#endif
  return (_dx & 1<<26) != 0;
}


int alignment_sse__compatible_proc() {
#if defined(__x86_64__)
  uint64_t _ax = 0, _bx = 0, _cx = 0, _dx = 0;
  asm volatile (
                "pushq %%rbx" "\n\t"
                "cpuid" "\n\t"
                "movq %%rbx,%1" "\n\t"
                "popq %%rbx" "\n\t"
                : "=a" (_ax), "=r" (_bx), "=c" (_cx), "=d" (_dx)
                : "a" (1)
                );
#else
  uint32_t _ax = 0, _bx = 0, _cx = 0, _dx = 0;
  asm volatile (
                "pushl %%ebx" "\n\t"
                "cpuid" "\n\t"
                "movl %%ebx,%1" "\n\t"
                "popl %%ebx" "\n\t"
                : "=a" (_ax), "=r" (_bx), "=c" (_cx), "=d" (_dx)
                : "a" (1)
                );
#endif
#ifdef DEBUG_SIMD
  fprintf(stderr,"[compatible sse ? %s]\n", _dx & 1<<25 ? "yes":"no");
#endif
  return (_dx & 1<<25) != 0;
}


/*
 * macros and global variables (no need to export because not usefull ouside this .c file)
 */


/* avx512bw macros */

#ifdef __AVX512BW__

#define NEXTREADSEQ_TRIA512(inout_byte,inout_nbnuc,                                                                                        \
                            inout_vector,out_vtype_vLA) {                                                                                  \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI16_TYPE(_mm512_set1)(*((uint16_t *)(inout_byte)));                                                               \
      inout_nbnuc    = 8;                                                                                                                  \
      inout_byte    += 2;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                                \
    inout_vector.v = EPI16_TYPE(_mm512_srli)(inout_vector.v,2);                                                                            \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTREADSEQ_HEXA512(inout_byte,inout_nbnuc,                                                                                        \
                            inout_vector,out_vtype_vLA) {                                                                                  \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI32_TYPE(_mm512_set1)(*((uint32_t *)(inout_byte)));                                                               \
      inout_nbnuc    = 16;                                                                                                                 \
      inout_byte    += 4;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                                \
    inout_vector.v = EPI32_TYPE(_mm512_srli)(inout_vector.v,2);                                                                            \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTREADSEQ_OCTA512(inout_byte,inout_nbnuc,                                                                                        \
                            inout_vector,out_vtype_vLA) {                                                                                  \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI64_TYPE(_mm512_set1)(*((uint64_t *)(inout_byte)));                                                               \
      inout_nbnuc    = 32;                                                                                                                 \
      inout_byte    += 8;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                                \
    inout_vector.v = EPI64_TYPE(_mm512_srli)(inout_vector.v,2);                                                                            \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTREADSEQ_QUAD512(inout_byte,inout_nbnuc,                                                                                        \
                            inout_vector,out_vtype_vLA) {                                                                                  \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI64_TYPE(_mm512_set1)(*((uint64_t *)(inout_byte)));                                                               \
      inout_nbnuc    = 32;                                                                                                                 \
      inout_byte    += 8;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                                \
    inout_vector.v = EPI64_TYPE(_mm512_srli)(inout_vector.v,2);                                                                            \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTGENOSEQ_TRIA512(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                               \
    __mmask32 __u__;                                                                                                                       \
    __u__ = EPI16_MASK_TYPE(_mm512_cmpeq)(inout_nbnuc_vector.v,SI512_TYPE(_mm512_setzero)());                                              \
    if (__u__) {                                                                                                                           \
      __mmask32 __d__;                                                                                                                     \
      for (__d__ = 0; __d__ < 32; __d__++) {                                                                                               \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          uint16_t i_nuc   = *((uint16_t *)(inout_byte[__d__]));                                                                           \
          uint16_t i_nbnuc = 8;                                                                                                            \
          if (inout_subpos[__d__]) {                                                                                                       \
            i_nuc              >>= inout_subpos[__d__] << 1;                                                                               \
            i_nbnuc             -= inout_subpos[__d__];                                                                                    \
            inout_subpos[__d__]  = 0;                                                                                                      \
          }                                                                                                                                \
          inout_vector.v       = EPI16_TYPE(_mm512_mask_set1)(inout_vector.v,       1 << (__d__), i_nuc);                                  \
          inout_nbnuc_vector.v = EPI16_TYPE(_mm512_mask_set1)(inout_nbnuc_vector.v, 1 << (__d__), i_nbnuc);                                \
          inout_byte[__d__]   += 2;                                                                                                        \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                          \
    inout_vector.v       = EPI16_TYPE(_mm512_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI16_TYPE(_mm512_sub)(inout_nbnuc_vector.v,vOnes512);                                                          \
}

#define NEXTGENOSEQ_HEXA512(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                               \
    __mmask16 __u__;                                                                                                                       \
    __u__ = EPI32_MASK_TYPE(_mm512_cmpeq)(inout_nbnuc_vector.v,SI512_TYPE(_mm512_setzero)());                                              \
    if (__u__) {                                                                                                                           \
      __mmask16 __d__;                                                                                                                     \
      for (__d__ = 0; __d__ < 16; __d__++) {                                                                                               \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          uint32_t i_nuc   = *((uint32_t *)(inout_byte[__d__]));                                                                           \
          uint32_t i_nbnuc = 16;                                                                                                           \
          if (inout_subpos[__d__]) {                                                                                                       \
            i_nuc              >>= inout_subpos[__d__] << 1;                                                                               \
            i_nbnuc             -= inout_subpos[__d__];                                                                                    \
            inout_subpos[__d__]  = 0;                                                                                                      \
          }                                                                                                                                \
          inout_vector.v       = EPI32_TYPE(_mm512_mask_set1)(inout_vector.v,       1 << (__d__), i_nuc);                                  \
          inout_nbnuc_vector.v = EPI32_TYPE(_mm512_mask_set1)(inout_nbnuc_vector.v, 1 << (__d__), i_nbnuc);                                \
          inout_byte[__d__]   += 4;                                                                                                        \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                          \
    inout_vector.v       = EPI32_TYPE(_mm512_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm512_sub)(inout_nbnuc_vector.v,vOnes512);                                                          \
}

#define NEXTGENOSEQ_OCTA512(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                               \
    __mmask8 __u__;                                                                                                                        \
    __u__ = EPI64_MASK_TYPE(_mm512_cmpeq)(inout_nbnuc_vector.v,SI512_TYPE(_mm512_setzero)());                                              \
    if (__u__) {                                                                                                                           \
     __mmask8 __d__;                                                                                                                       \
      for (__d__ = 0; __d__ < 8; __d__++) {                                                                                                \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          uint64_t i_nuc   = *((uint64_t *)(inout_byte[__d__]));                                                                           \
          uint64_t i_nbnuc = 32;                                                                                                           \
          if (inout_subpos[__d__]) {                                                                                                       \
            i_nuc              >>= inout_subpos[__d__] << 1;                                                                               \
            i_nbnuc             -= inout_subpos[__d__];                                                                                    \
            inout_subpos[__d__]  = 0;                                                                                                      \
          }                                                                                                                                \
          inout_vector.v       = EPI64_TYPE(_mm512_mask_set1)(inout_vector.v,       1 << (__d__), i_nuc);                                  \
          inout_nbnuc_vector.v = EPI64_TYPE(_mm512_mask_set1)(inout_nbnuc_vector.v, 1 << (__d__), i_nbnuc);                                \
          inout_byte[__d__]   += 8;                                                                                                        \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                          \
    inout_vector.v       = EPI64_TYPE(_mm512_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI64_TYPE(_mm512_sub)(inout_nbnuc_vector.v,vOnes512);                                                          \
}

#define NEXTGENOSEQ_QUAD512(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                               \
    __mmask8 __u__;                                                                                                                        \
    __u__ = EPI64_MASK_TYPE(_mm512_cmpeq)(inout_nbnuc_vector.v,SI512_TYPE(_mm512_setzero)());                                              \
    if (__u__) {                                                                                                                           \
     __mmask8 __d__;                                                                                                                       \
      for (__d__ = 0; __d__ < 4; __d__++) {                                                                                                \
        if (__u__ & 3 << (2*__d__)) {                                                                                                      \
          uint64_t i_nuc   = *((uint64_t *)(inout_byte[__d__]));                                                                           \
          uint64_t i_nbnuc = 32;                                                                                                           \
          if (inout_subpos[__d__]) {                                                                                                       \
            i_nuc              >>= inout_subpos[__d__] << 1;                                                                               \
            i_nbnuc             -= inout_subpos[__d__];                                                                                    \
            inout_subpos[__d__]  = 0;                                                                                                      \
          }                                                                                                                                \
          inout_vector.v       = EPI64_TYPE(_mm512_mask_set1)(inout_vector.v,       3 << (2*__d__), i_nuc);                                \
          inout_nbnuc_vector.v = EPI64_TYPE(_mm512_mask_set1)(inout_nbnuc_vector.v, 3 << (2*__d__), i_nbnuc);                              \
          inout_byte[__d__]   += 8;                                                                                                        \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                          \
    inout_vector.v       = EPI64_TYPE(_mm512_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI64_TYPE(_mm512_sub)(inout_nbnuc_vector.v,vOnes512);                                                          \
}

#define NEXTGENOSEQ_NOSUB_TRIA512(inout_byte,                                                                                              \
                                  inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                         \
    __mmask32 __u__;                                                                                                                       \
    __u__ = EPI16_MASK_TYPE(_mm512_cmpeq)(inout_nbnuc_vector.v,SI512_TYPE(_mm512_setzero)());                                              \
    if (__u__) {                                                                                                                           \
      __mmask32 __d__;                                                                                                                     \
      for (__d__ = 0; __d__ < 32; __d__++) {                                                                                               \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          uint16_t i_nuc   = *((uint16_t *)(inout_byte[__d__]));                                                                           \
          uint16_t i_nbnuc = 8;                                                                                                            \
          inout_vector.v       = EPI16_TYPE(_mm512_mask_set1)(inout_vector.v,       1 << (__d__), i_nuc);                                  \
          inout_nbnuc_vector.v = EPI16_TYPE(_mm512_mask_set1)(inout_nbnuc_vector.v, 1 << (__d__), i_nbnuc);                                \
          inout_byte[__d__]   += 2;                                                                                                        \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                          \
    inout_vector.v       = EPI16_TYPE(_mm512_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI16_TYPE(_mm512_sub)(inout_nbnuc_vector.v,vOnes512);                                                          \
}

#define NEXTGENOSEQ_NOSUB_HEXA512(inout_byte,                                                                                              \
                                  inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                         \
    __mmask16 __u__;                                                                                                                       \
    __u__ = EPI32_MASK_TYPE(_mm512_cmpeq)(inout_nbnuc_vector.v,SI512_TYPE(_mm512_setzero)());                                              \
    if (__u__) {                                                                                                                           \
      __mmask16 __d__;                                                                                                                     \
      for (__d__ = 0; __d__ < 16; __d__++) {                                                                                               \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          uint32_t i_nuc   = *((uint32_t *)(inout_byte[__d__]));                                                                           \
          uint32_t i_nbnuc = 16;                                                                                                           \
          inout_vector.v       = EPI32_TYPE(_mm512_mask_set1)(inout_vector.v,       1 << (__d__), i_nuc);                                  \
          inout_nbnuc_vector.v = EPI32_TYPE(_mm512_mask_set1)(inout_nbnuc_vector.v, 1 << (__d__), i_nbnuc);                                \
          inout_byte[__d__]   += 4;                                                                                                        \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                          \
    inout_vector.v       = EPI32_TYPE(_mm512_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm512_sub)(inout_nbnuc_vector.v,vOnes512);                                                          \
}

#define NEXTGENOSEQ_NOSUB_OCTA512(inout_byte,                                                                                              \
                                  inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                         \
    __mmask8 __u__;                                                                                                                        \
    __u__ = EPI64_MASK_TYPE(_mm512_cmpeq)(inout_nbnuc_vector.v,SI512_TYPE(_mm512_setzero)());                                              \
    if (__u__) {                                                                                                                           \
     __mmask8 __d__;                                                                                                                       \
      for (__d__ = 0; __d__ < 8; __d__++) {                                                                                                \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          uint64_t i_nuc   = *((uint64_t *)(inout_byte[__d__]));                                                                           \
          uint64_t i_nbnuc = 32;                                                                                                           \
          inout_vector.v       = EPI64_TYPE(_mm512_mask_set1)(inout_vector.v,       1 << (__d__), i_nuc);                                  \
          inout_nbnuc_vector.v = EPI64_TYPE(_mm512_mask_set1)(inout_nbnuc_vector.v, 1 << (__d__), i_nbnuc);                                \
          inout_byte[__d__]   += 8;                                                                                                        \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                          \
    inout_vector.v       = EPI64_TYPE(_mm512_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI64_TYPE(_mm512_sub)(inout_nbnuc_vector.v,vOnes512);                                                          \
}

#define NEXTGENOSEQ_NOSUB_QUAD512(inout_byte,                                                                                              \
                                  inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                         \
    __mmask8 __u__;                                                                                                                        \
    __u__ = EPI64_MASK_TYPE(_mm512_cmpeq)(inout_nbnuc_vector.v,SI512_TYPE(_mm512_setzero)());                                              \
    if (__u__) {                                                                                                                           \
     __mmask8 __d__;                                                                                                                       \
      for (__d__ = 0; __d__ < 4; __d__++) {                                                                                                \
        if (__u__ & 3 << (2*__d__)) {                                                                                                      \
          uint64_t i_nuc   = *((uint64_t *)(inout_byte[__d__]));                                                                           \
          uint64_t i_nbnuc = 32;                                                                                                           \
          inout_vector.v       = EPI64_TYPE(_mm512_mask_set1)(inout_vector.v,       3 << (2*__d__), i_nuc);                                \
          inout_nbnuc_vector.v = EPI64_TYPE(_mm512_mask_set1)(inout_nbnuc_vector.v, 3 << (2*__d__), i_nbnuc);                              \
          inout_byte[__d__]   += 8;                                                                                                        \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                          \
    inout_vector.v       = EPI64_TYPE(_mm512_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI64_TYPE(_mm512_sub)(inout_nbnuc_vector.v,vOnes512);                                                          \
}

#define NEXTGENOSEQ_NOSUB_NOUP_TRIA512(inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                    \
    out_vtype_vLB        = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                          \
    inout_vector.v       = EPI16_TYPE(_mm512_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI16_TYPE(_mm512_sub)(inout_nbnuc_vector.v,vOnes512);                                                          \
}

#define NEXTGENOSEQ_NOSUB_NOUP_HEXA512(inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                    \
    out_vtype_vLB        = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                          \
    inout_vector.v       = EPI32_TYPE(_mm512_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm512_sub)(inout_nbnuc_vector.v,vOnes512);                                                          \
}

#define NEXTGENOSEQ_NOSUB_NOUP_OCTA512(inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                    \
    out_vtype_vLB        = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                          \
    inout_vector.v       = EPI64_TYPE(_mm512_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI64_TYPE(_mm512_sub)(inout_nbnuc_vector.v,vOnes512);                                                          \
}

#define NEXTGENOSEQ_NOSUB_NOUP_QUAD512(inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                    \
    out_vtype_vLB        = SI512_TYPE(_mm512_and)(inout_vector.v,vBufferMask512);                                                          \
    inout_vector.v       = EPI64_TYPE(_mm512_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI64_TYPE(_mm512_sub)(inout_nbnuc_vector.v,vOnes512);                                                          \
}

VTYPE512  vThreshold512     __attribute__ ((aligned (64))),
          vMatchS512        __attribute__ ((aligned (64))),
          vMismatchS512     __attribute__ ((aligned (64))),
          vIndelOpenS512    __attribute__ ((aligned (64))),
          vIndelExtendsS512 __attribute__ ((aligned (64))),
          vOnes512          __attribute__ ((aligned (64))),
          vBufferMask512    __attribute__ ((aligned (64))),
         *vMsk512           __attribute__ ((aligned (64)));

void     *vMsk512unaligned = NULL;

void alignment_avx512bw__clean() {if (vMsk512unaligned){ free(vMsk512unaligned); vMsk512unaligned = NULL;}}

#endif




/* avx2 macros */

#ifdef __AVX2__

#define NEXTREADSEQ_HEXA256(inout_byte,inout_nbnuc,                                                                                        \
                            inout_vector,out_vtype_vLA) {                                                                                  \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI16_TYPE(_mm256_set1)(*((uint16_t *)(inout_byte)));                                                               \
      inout_nbnuc    = 8;                                                                                                                  \
      inout_byte    += 2;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                                \
    inout_vector.v = EPI16_TYPE(_mm256_srli)(inout_vector.v,2);                                                                            \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTREADSEQ_OCTA256(inout_byte,inout_nbnuc,                                                                                        \
                            inout_vector,out_vtype_vLA) {                                                                                  \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI32_TYPE(_mm256_set1)(*((uint32_t *)(inout_byte)));                                                               \
      inout_nbnuc    = 16;                                                                                                                 \
      inout_byte    += 4;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                                \
    inout_vector.v = EPI32_TYPE(_mm256_srli)(inout_vector.v,2);                                                                            \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTREADSEQ_QUAD256(inout_byte,inout_nbnuc,                                                                                        \
                            inout_vector,out_vtype_vLA) {                                                                                  \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI64X_TYPE(_mm256_set1)(*((uint64_t *)(inout_byte)));                                                              \
      inout_nbnuc    = 32;                                                                                                                 \
      inout_byte    += 8;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                                \
    inout_vector.v = EPI64_TYPE(_mm256_srli)(inout_vector.v,2);                                                                            \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTREADSEQ_PAIR256(inout_byte,inout_nbnuc,                                                                                        \
                            inout_vector,out_vtype_vLA) {                                                                                  \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI64X_TYPE(_mm256_set1)(*((uint64_t *)(inout_byte)));                                                              \
      inout_nbnuc    = 32;                                                                                                                 \
      inout_byte    += 8;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                                \
    inout_vector.v = EPI64_TYPE(_mm256_srli)(inout_vector.v,2);                                                                            \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTGENOSEQ_HEXA256(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                               \
    unsigned int __u__;                                                                                                                    \
    __u__ = EPI8_TYPE(_mm256_movemask)(EPI16_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                       \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 16; __d__++) {                                                                                               \
        if (__u__ & 1 << (2*__d__)) {                                                                                                      \
          inout_vector.u16[__d__] =                                                                                                        \
            *((uint16_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u16[__d__] = 8;                                                                                               \
          inout_byte[__d__]            += 2;                                                                                               \
          if (inout_subpos[__d__]) {                                                                                                       \
            inout_vector.u16[__d__]      >>= inout_subpos[__d__] << 1;                                                                     \
            inout_nbnuc_vector.u16[__d__] -= inout_subpos[__d__];                                                                          \
            inout_subpos[__d__]            = 0;                                                                                            \
          }                                                                                                                                \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                          \
    inout_vector.v       = EPI16_TYPE(_mm256_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI16_TYPE(_mm256_sub)(inout_nbnuc_vector.v,vOnes256);                                                          \
}

#define NEXTGENOSEQ_OCTA256(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                               \
    unsigned int __u__;                                                                                                                    \
    __u__ = _mm256_movemask_ps((__m256)EPI32_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                       \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 8; __d__++) {                                                                                                \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          inout_vector.u32[__d__] =                                                                                                        \
            *((uint32_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u32[__d__] = 16;                                                                                              \
          inout_byte[__d__]            += 4;                                                                                               \
          if (inout_subpos[__d__]) {                                                                                                       \
            inout_vector.u32[__d__]      >>= inout_subpos[__d__] << 1;                                                                     \
            inout_nbnuc_vector.u32[__d__] -= inout_subpos[__d__];                                                                          \
            inout_subpos[__d__]            = 0;                                                                                            \
          }                                                                                                                                \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                          \
    inout_vector.v       = EPI32_TYPE(_mm256_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm256_sub)(inout_nbnuc_vector.v,vOnes256);                                                          \
}

#define NEXTGENOSEQ_QUAD256(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                               \
    unsigned int __u__;                                                                                                                    \
    __u__ = _mm256_movemask_pd((__m256d)EPI64_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                      \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 4; __d__++) {                                                                                                \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          inout_vector.u64[__d__] =                                                                                                        \
            *((uint64_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u64[__d__] = 32;                                                                                              \
          inout_byte[__d__]            += 8;                                                                                               \
          if (inout_subpos[__d__]) {                                                                                                       \
            inout_vector.u64[__d__]      >>= inout_subpos[__d__] << 1;                                                                     \
            inout_nbnuc_vector.u64[__d__] -= inout_subpos[__d__];                                                                          \
            inout_subpos[__d__]            = 0;                                                                                            \
          }                                                                                                                                \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                          \
    inout_vector.v       = EPI64_TYPE(_mm256_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI64_TYPE(_mm256_sub)(inout_nbnuc_vector.v,vOnes256);                                                          \
}

#define NEXTGENOSEQ_PAIR256(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                               \
    unsigned int __u__;                                                                                                                    \
    __u__ = _mm256_movemask_pd((__m256d)EPI64_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                      \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 2; __d__++) {                                                                                                \
        if (__u__ & 1 << (2*__d__)) {                                                                                                      \
          inout_vector.u64[2*__d__] =                                                                                                      \
            *((uint64_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u64[2*__d__    ] = 32;                                                                                        \
          inout_nbnuc_vector.u64[2*__d__ + 1] = 32;                                                                                        \
          inout_byte[__d__]                  += 8;                                                                                         \
          if (inout_subpos[__d__]) {                                                                                                       \
            inout_vector.u64[2*__d__]          >>= inout_subpos[__d__] << 1;                                                               \
            inout_nbnuc_vector.u64[2*__d__    ] -= inout_subpos[__d__];                                                                    \
            inout_nbnuc_vector.u64[2*__d__ + 1] -= inout_subpos[__d__];                                                                    \
            inout_subpos[__d__]                  = 0;                                                                                      \
          }                                                                                                                                \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                          \
    inout_vector.v       = EPI64_TYPE(_mm256_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI64_TYPE(_mm256_sub)(inout_nbnuc_vector.v,vOnes256);                                                          \
}

#define NEXTGENOSEQ_NOSUB_HEXA256(inout_byte,                                                                                              \
                                  inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                         \
    unsigned int __u__;                                                                                                                    \
    __u__ = EPI8_TYPE(_mm256_movemask)(EPI16_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                       \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 16; __d__++) {                                                                                               \
        if (__u__ & 1 << (2*__d__)) {                                                                                                      \
          inout_vector.u16[__d__] =                                                                                                        \
            *((uint16_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u16[__d__] = 8;                                                                                               \
          inout_byte[__d__]            += 2;                                                                                               \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                          \
    inout_vector.v       = EPI16_TYPE(_mm256_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI16_TYPE(_mm256_sub)(inout_nbnuc_vector.v,vOnes256);                                                          \
}

#define NEXTGENOSEQ_NOSUB_OCTA256(inout_byte,                                                                                              \
                                  inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                         \
    unsigned int __u__;                                                                                                                    \
    __u__ = _mm256_movemask_ps((__m256)EPI32_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                       \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 8; __d__++) {                                                                                                \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          inout_vector.u32[__d__] =                                                                                                        \
            *((uint32_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u32[__d__] = 16;                                                                                              \
          inout_byte[__d__]            += 4;                                                                                               \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                          \
    inout_vector.v       = EPI32_TYPE(_mm256_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm256_sub)(inout_nbnuc_vector.v,vOnes256);                                                          \
}

#define NEXTGENOSEQ_NOSUB_QUAD256(inout_byte,                                                                                              \
                                  inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                         \
    unsigned int __u__;                                                                                                                    \
    __u__ = _mm256_movemask_pd((__m256d)EPI64_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                      \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 4; __d__++) {                                                                                                \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          inout_vector.u64[__d__] =                                                                                                        \
            *((uint64_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u64[__d__] = 32;                                                                                              \
          inout_byte[__d__]            += 8;                                                                                               \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                          \
    inout_vector.v       = EPI64_TYPE(_mm256_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI64_TYPE(_mm256_sub)(inout_nbnuc_vector.v,vOnes256);                                                          \
}

#define NEXTGENOSEQ_NOSUB_PAIR256(inout_byte,                                                                                              \
                                  inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                         \
    unsigned int __u__;                                                                                                                    \
    __u__ = _mm256_movemask_pd((__m256d)EPI64_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                      \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 2; __d__++) {                                                                                                \
        if (__u__ & 1 << (2*__d__)) {                                                                                                      \
          inout_vector.u64[2*__d__] =                                                                                                      \
            *((uint64_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u64[2*__d__    ] = 32;                                                                                        \
          inout_nbnuc_vector.u64[2*__d__ + 1] = 32;                                                                                        \
          inout_byte[__d__]                  += 8;                                                                                         \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                          \
    inout_vector.v       = EPI64_TYPE(_mm256_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI64_TYPE(_mm256_sub)(inout_nbnuc_vector.v,vOnes256);                                                          \
}

#define NEXTGENOSEQ_NOSUB_NOUP_HEXA256(inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                    \
    out_vtype_vLB        = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                          \
    inout_vector.v       = EPI16_TYPE(_mm256_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI16_TYPE(_mm256_sub)(inout_nbnuc_vector.v,vOnes256);                                                          \
}

#define NEXTGENOSEQ_NOSUB_NOUP_OCTA256(inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                    \
    out_vtype_vLB        = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                          \
    inout_vector.v       = EPI32_TYPE(_mm256_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm256_sub)(inout_nbnuc_vector.v,vOnes256);                                                          \
}

#define NEXTGENOSEQ_NOSUB_NOUP_QUAD256(inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                    \
    out_vtype_vLB        = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                          \
    inout_vector.v       = EPI64_TYPE(_mm256_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI64_TYPE(_mm256_sub)(inout_nbnuc_vector.v,vOnes256);                                                          \
}

#define NEXTGENOSEQ_NOSUB_NOUP_PAIR256(inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                    \
    out_vtype_vLB        = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                          \
    inout_vector.v       = EPI64_TYPE(_mm256_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI64_TYPE(_mm256_sub)(inout_nbnuc_vector.v,vOnes256);                                                          \
}

VTYPE256  vThreshold256     __attribute__ ((aligned (32))),
          vMatchS256        __attribute__ ((aligned (32))),
          vMismatchS256     __attribute__ ((aligned (32))),
          vIndelOpenS256    __attribute__ ((aligned (32))),
          vIndelExtendsS256 __attribute__ ((aligned (32))),
          vOnes256          __attribute__ ((aligned (32))),
          vBufferMask256    __attribute__ ((aligned (32))),
         *vMsk256           __attribute__ ((aligned (32)));

void     *vMsk256unaligned = NULL;

void alignment_avx2__clean() {if (vMsk256unaligned){ free(vMsk256unaligned); vMsk256unaligned = NULL;}}

#endif




/* sse2 macros */

#ifdef __SSE2__

#define NEXTREADSEQ_OCTA128(inout_byte,inout_nbnuc,                                                                                        \
                            inout_vector,out_vtype_vLA) {                                                                                  \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI16_TYPE(_mm_set1)(*(uint16_t *)(inout_byte));                                                                    \
      inout_nbnuc    = 8;                                                                                                                  \
      inout_byte    += 2;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                                   \
    inout_vector.v = EPI16_TYPE(_mm_srli)(inout_vector.v,2);                                                                               \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTREADSEQ_QUAD128(inout_byte,inout_nbnuc,                                                                                        \
                            inout_vector,out_vtype_vLA) {                                                                                  \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI32_TYPE(_mm_set1)(*(uint32_t *)(inout_byte));                                                                    \
      inout_nbnuc    = 16;                                                                                                                 \
      inout_byte    += 4;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                                   \
    inout_vector.v = EPI32_TYPE(_mm_srli)(inout_vector.v,2);                                                                               \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTREADSEQ_PAIR128(inout_byte,inout_nbnuc,                                                                                        \
                            inout_vector,out_vtype_vLA) {                                                                                  \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI64X_TYPE(_mm_set1)(*(uint64_t *)(inout_byte));                                                                   \
      inout_nbnuc    = 32;                                                                                                                 \
      inout_byte    += 8;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                                   \
    inout_vector.v = EPI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                               \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTREADSEQ_MONO128(inout_byte,                                                                                                    \
                            inout_nbnuc,inout_vector,out_vtype_vLA) {                                                                      \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI64X_TYPE(_mm_set1)(*(uint64_t *)(inout_byte));                                                                   \
      inout_nbnuc    = 32;                                                                                                                 \
      inout_byte    += 8;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                                   \
    inout_vector.v = EPI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                               \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTGENOSEQ_OCTA128(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                               \
    unsigned int __u__;                                                                                                                    \
    __u__ = EPI8_TYPE(_mm_movemask)(EPI16_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI128_TYPE(_mm_setzero)()));                                \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 8; __d__++) {                                                                                                \
        if (__u__ & 1 << (2*__d__)) {                                                                                                      \
          inout_vector.u16[__d__] =                                                                                                        \
            *((uint16_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u16[__d__] = 8;                                                                                               \
          inout_byte[__d__]            += 2;                                                                                               \
          if (inout_subpos[__d__]) {                                                                                                       \
            inout_vector.u16[__d__]      >>= inout_subpos[__d__] << 1;                                                                     \
            inout_nbnuc_vector.u16[__d__] -= inout_subpos[__d__];                                                                          \
            inout_subpos[__d__]            = 0;                                                                                            \
          }                                                                                                                                \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                             \
    inout_vector.v       = EPI16_TYPE(_mm_srli)(inout_vector.v,2);                                                                         \
    inout_nbnuc_vector.v = EPI16_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes128);                                                             \
}

#define NEXTGENOSEQ_QUAD128(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                               \
    unsigned int __u__;                                                                                                                    \
    __u__ = _mm_movemask_ps((__m128)EPI32_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI128_TYPE(_mm_setzero)()));                                \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 4; __d__++) {                                                                                                \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          inout_vector.u32[__d__] =                                                                                                        \
            *((uint32_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u32[__d__] = 16;                                                                                              \
          inout_byte[__d__]            += 4;                                                                                               \
          if (inout_subpos[__d__]) {                                                                                                       \
            inout_vector.u32[__d__]      >>= inout_subpos[__d__] << 1;                                                                     \
            inout_nbnuc_vector.u32[__d__] -= inout_subpos[__d__];                                                                          \
            inout_subpos[__d__]            = 0;                                                                                            \
          }                                                                                                                                \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                             \
    inout_vector.v       = EPI32_TYPE(_mm_srli)(inout_vector.v,2);                                                                         \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes128);                                                             \
}

#define NEXTGENOSEQ_PAIR128(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                               \
    unsigned int __u__;                                                                                                                    \
    __u__ = _mm_movemask_pd((__m128d)EPI32_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI128_TYPE(_mm_setzero)()));                               \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 2; __d__++) {                                                                                                \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          inout_vector.u64[__d__] =                                                                                                        \
            *((uint64_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u32[2*__d__    ] = 32;                                                                                        \
          inout_nbnuc_vector.u32[2*__d__ + 1] = 32;                                                                                        \
          inout_byte[__d__]                  += 8;                                                                                         \
          if (inout_subpos[__d__]) {                                                                                                       \
            inout_vector.u64[__d__]            >>= inout_subpos[__d__] << 1;                                                               \
            inout_nbnuc_vector.u32[2*__d__    ] -= inout_subpos[__d__];                                                                    \
            inout_nbnuc_vector.u32[2*__d__ + 1] -= inout_subpos[__d__];                                                                    \
            inout_subpos[__d__]                  = 0;                                                                                      \
          }                                                                                                                                \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                             \
    inout_vector.v       = EPI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                         \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes128);                                                             \
}

#define NEXTGENOSEQ_MONO128(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc,inout_vector,out_vtype_vLB) {                                                                      \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI64X_TYPE(_mm_set1)(*((uint64_t *)(inout_byte)));                                                                 \
      inout_nbnuc    = 32;                                                                                                                 \
      inout_byte    += 8;                                                                                                                  \
      if (inout_subpos) {                                                                                                                  \
        inout_vector.v  = EPI64_TYPE(_mm_srli)(inout_vector.v, inout_subpos << 1);                                                         \
        inout_nbnuc    -= inout_subpos;                                                                                                    \
        inout_subpos    = 0;                                                                                                               \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                             \
    inout_vector.v       = EPI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                         \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTGENOSEQ_NOSUB_OCTA128(inout_byte,                                                                                              \
                                  inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                         \
    unsigned int __u__;                                                                                                                    \
    __u__ = EPI8_TYPE(_mm_movemask)(EPI16_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI128_TYPE(_mm_setzero)()));                                \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 8; __d__++) {                                                                                                \
        if (__u__ & 1 << (2*__d__)) {                                                                                                      \
          inout_vector.u16[__d__] =                                                                                                        \
            *((uint16_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u16[__d__] = 8;                                                                                               \
          inout_byte[__d__]            += 2;                                                                                               \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                             \
    inout_vector.v       = EPI16_TYPE(_mm_srli)(inout_vector.v,2);                                                                         \
    inout_nbnuc_vector.v = EPI16_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes128);                                                             \
}

#define NEXTGENOSEQ_NOSUB_QUAD128(inout_byte,                                                                                              \
                                  inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                         \
    unsigned int __u__;                                                                                                                    \
    __u__ = _mm_movemask_ps((__m128)EPI32_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI128_TYPE(_mm_setzero)()));                                \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 4; __d__++) {                                                                                                \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          inout_vector.u32[__d__] =                                                                                                        \
            *((uint32_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u32[__d__] = 16;                                                                                              \
          inout_byte[__d__]            += 4;                                                                                               \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                             \
    inout_vector.v       = EPI32_TYPE(_mm_srli)(inout_vector.v,2);                                                                         \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes128);                                                             \
}

#define NEXTGENOSEQ_NOSUB_PAIR128(inout_byte,                                                                                              \
                                  inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                         \
    unsigned int __u__;                                                                                                                    \
    __u__ = _mm_movemask_pd((__m128d)EPI32_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI128_TYPE(_mm_setzero)()));                               \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 2; __d__++) {                                                                                                \
        if (__u__ & 1 << (__d__)) {                                                                                                        \
          inout_vector.u64[__d__] =                                                                                                        \
            *((uint64_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u32[2*__d__    ] = 32;                                                                                        \
          inout_nbnuc_vector.u32[2*__d__ + 1] = 32;                                                                                        \
          inout_byte[__d__]                  += 8;                                                                                         \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                             \
    inout_vector.v       = EPI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                         \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes128);                                                             \
}

#define NEXTGENOSEQ_NOSUB_MONO128(inout_byte,                                                                                              \
                                  inout_nbnuc,inout_vector,out_vtype_vLB) {                                                                \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = EPI64X_TYPE(_mm_set1)(*((uint64_t *)(inout_byte)));                                                                 \
      inout_nbnuc    = 32;                                                                                                                 \
      inout_byte    += 8;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLB        = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                             \
    inout_vector.v       = EPI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                         \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTGENOSEQ_NOSUB_NOUP_OCTA128(inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                    \
    out_vtype_vLB        = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                             \
    inout_vector.v       = EPI16_TYPE(_mm_srli)(inout_vector.v,2);                                                                         \
    inout_nbnuc_vector.v = EPI16_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes128);                                                             \
}

#define NEXTGENOSEQ_NOSUB_NOUP_QUAD128(inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                    \
    out_vtype_vLB        = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                             \
    inout_vector.v       = EPI32_TYPE(_mm_srli)(inout_vector.v,2);                                                                         \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes128);                                                             \
}

#define NEXTGENOSEQ_NOSUB_NOUP_PAIR128(inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                    \
    out_vtype_vLB        = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                             \
    inout_vector.v       = EPI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                         \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes128);                                                             \
}

#define NEXTGENOSEQ_NOSUB_NOUP_MONO128(inout_nbnuc,inout_vector,out_vtype_vLB) {                                                           \
    out_vtype_vLB        = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                             \
    inout_vector.v       = EPI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                         \
    inout_nbnuc--;                                                                                                                         \
}

VTYPE128  vThreshold128     __attribute__ ((aligned (16))),
          vMatchS128        __attribute__ ((aligned (16))),
          vMismatchS128     __attribute__ ((aligned (16))),
          vIndelOpenS128    __attribute__ ((aligned (16))),
          vIndelExtendsS128 __attribute__ ((aligned (16))),
          vOnes128          __attribute__ ((aligned (16))),
          vBufferMask128    __attribute__ ((aligned (16))),
         *vMsk128           __attribute__ ((aligned (16)));

void     *vMsk128unaligned = NULL;

void alignment_sse2__clean() {if (vMsk128unaligned){ free(vMsk128unaligned); vMsk128unaligned = NULL;}}

#endif




/* sse macros */

#ifdef __SSE__

#define NEXTREADSEQ_QUAD64(inout_byte,inout_nbnuc,                                                                                         \
                           inout_vector,out_vtype_vLA) {                                                                                   \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = PI16_TYPE(_mm_set1)(*(uint16_t *)(inout_byte));                                                                     \
      inout_nbnuc    = 8;                                                                                                                  \
      inout_byte    += 2;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA        = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                               \
    inout_vector.v       = PI16_TYPE(_mm_srli)(inout_vector.v,2);                                                                          \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTREADSEQ_PAIR64(inout_byte,inout_nbnuc,                                                                                         \
                           inout_vector,out_vtype_vLA) {                                                                                   \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = PI32_TYPE(_mm_set1)(*(uint32_t *)(inout_byte));                                                                     \
      inout_nbnuc    = 16;                                                                                                                 \
      inout_byte    += 4;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA        = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                               \
    inout_vector.v       = PI32_TYPE(_mm_srli)(inout_vector.v,2);                                                                          \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTREADSEQ_MONO64(inout_byte,inout_nbnuc,                                                                                         \
                           inout_vector,out_vtype_vLA) {                                                                                   \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = _mm_cvtsi64_m64(*((uint64_t *)(inout_byte)));                                                                       \
      inout_nbnuc    = 32;                                                                                                                 \
      inout_byte    += 8;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA        = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                               \
    inout_vector.v       = SI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                          \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTGENOSEQ_QUAD64(inout_subpos,inout_byte,                                                                                        \
                           inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                                \
    unsigned int __u__;                                                                                                                    \
    __u__ = PI8_TYPE(_mm_movemask)(PI16_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI64_TYPE(_mm_setzero)()));                                   \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 4; __d__++) {                                                                                                \
        if (__u__ & 1 << (2*__d__)) {                                                                                                      \
          inout_vector.u16[__d__] =                                                                                                        \
            *((uint16_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u16[__d__] = 8;                                                                                               \
          inout_byte[__d__]            += 2;                                                                                               \
          if (inout_subpos[__d__]) {                                                                                                       \
            inout_vector.u16[__d__]      >>= inout_subpos[__d__] << 1;                                                                     \
            inout_nbnuc_vector.u16[__d__] -= inout_subpos[__d__];                                                                          \
            inout_subpos[__d__]            = 0;                                                                                            \
          }                                                                                                                                \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                               \
    inout_vector.v       = PI16_TYPE(_mm_srli)(inout_vector.v,2);                                                                          \
    inout_nbnuc_vector.v = PI16_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes64);                                                               \
}

#define NEXTGENOSEQ_PAIR64(inout_subpos,inout_byte,                                                                                        \
                           inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                                \
    unsigned int __u__;                                                                                                                    \
    __u__ = PI8_TYPE(_mm_movemask)(PI32_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI64_TYPE(_mm_setzero)()));                                   \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 2; __d__++) {                                                                                                \
        if (__u__ & 1 << (4*__d__)) {                                                                                                      \
          inout_vector.u32[__d__] =                                                                                                        \
            *((uint32_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u32[__d__] = 16;                                                                                              \
          inout_byte[__d__]            += 4;                                                                                               \
          if (inout_subpos[__d__]) {                                                                                                       \
            inout_vector.u32[__d__]      >>= inout_subpos[__d__] << 1;                                                                     \
            inout_nbnuc_vector.u32[__d__] -= inout_subpos[__d__];                                                                          \
            inout_subpos[__d__]            = 0;                                                                                            \
          }                                                                                                                                \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                               \
    inout_vector.v       = PI32_TYPE(_mm_srli)(inout_vector.v,2);                                                                          \
    inout_nbnuc_vector.v = PI32_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes64);                                                               \
}

#define NEXTGENOSEQ_MONO64(inout_subpos,inout_byte,                                                                                        \
                           inout_nbnuc,inout_vector,out_vtype_vLB) {                                                                       \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = _mm_cvtsi64_m64(*((uint64_t *)(inout_byte)));                                                                       \
      inout_nbnuc    = 32;                                                                                                                 \
      inout_byte    += 8;                                                                                                                  \
      if (inout_subpos) {                                                                                                                  \
        inout_vector.v = SI64_TYPE(_mm_srli)(inout_vector.v,inout_subpos << 1);                                                            \
        inout_nbnuc   -= inout_subpos;                                                                                                     \
        inout_subpos   = 0;                                                                                                                \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                               \
    inout_vector.v       = SI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                          \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTGENOSEQ_NOSUB_QUAD64(inout_byte,                                                                                               \
                                 inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                          \
    unsigned int __u__;                                                                                                                    \
    __u__ = PI8_TYPE(_mm_movemask)(PI16_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI64_TYPE(_mm_setzero)()));                                   \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 4; __d__++) {                                                                                                \
        if (__u__ & 1 << (2*__d__)) {                                                                                                      \
          inout_vector.u16[__d__] =                                                                                                        \
            *((uint16_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u16[__d__] = 8;                                                                                               \
          inout_byte[__d__]            += 2;                                                                                               \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                               \
    inout_vector.v       = PI16_TYPE(_mm_srli)(inout_vector.v,2);                                                                          \
    inout_nbnuc_vector.v = PI16_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes64);                                                               \
}

#define NEXTGENOSEQ_NOSUB_PAIR64(inout_byte,                                                                                               \
                                 inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                          \
    unsigned int __u__;                                                                                                                    \
    __u__ = PI8_TYPE(_mm_movemask)(PI32_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI64_TYPE(_mm_setzero)()));                                   \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 2; __d__++) {                                                                                                \
        if (__u__ & 1 << (4*__d__)) {                                                                                                      \
          inout_vector.u32[__d__] =                                                                                                        \
            *((uint32_t *)(inout_byte[__d__]));                                                                                            \
          inout_nbnuc_vector.u32[__d__] = 16;                                                                                              \
          inout_byte[__d__]            += 4;                                                                                               \
        }                                                                                                                                  \
      }                                                                                                                                    \
    }                                                                                                                                      \
    out_vtype_vLB        = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                               \
    inout_vector.v       = PI32_TYPE(_mm_srli)(inout_vector.v,2);                                                                          \
    inout_nbnuc_vector.v = PI32_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes64);                                                               \
}

#define NEXTGENOSEQ_NOSUB_MONO64(inout_byte,                                                                                               \
                                 inout_nbnuc,inout_vector,out_vtype_vLB) {                                                                 \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = _mm_cvtsi64_m64(*((uint64_t *)(inout_byte)));                                                                       \
      inout_nbnuc    = 32;                                                                                                                 \
      inout_byte    += 8;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLB        = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                               \
    inout_vector.v       = SI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                          \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTGENOSEQ_NOSUB_NOUP_QUAD64(inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                     \
    out_vtype_vLB        = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                               \
    inout_vector.v       = PI16_TYPE(_mm_srli)(inout_vector.v,2);                                                                          \
    inout_nbnuc_vector.v = PI16_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes64);                                                               \
}

#define NEXTGENOSEQ_NOSUB_NOUP_PAIR64(inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                     \
    out_vtype_vLB        = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                               \
    inout_vector.v       = PI32_TYPE(_mm_srli)(inout_vector.v,2);                                                                          \
    inout_nbnuc_vector.v = PI32_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes64);                                                               \
}

#define NEXTGENOSEQ_NOSUB_NOUP_MONO64(inout_nbnuc,inout_vector,out_vtype_vLB) {                                                            \
    out_vtype_vLB        = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                               \
    inout_vector.v       = SI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                          \
    inout_nbnuc--;                                                                                                                         \
}

VTYPE64  vThreshold64     __attribute__ ((aligned (16))),
         vMatchS64        __attribute__ ((aligned (16))),
         vMismatchS64     __attribute__ ((aligned (16))),
         vIndelOpenS64    __attribute__ ((aligned (16))),
         vIndelExtendsS64 __attribute__ ((aligned (16))),
         vOnes64          __attribute__ ((aligned (16))),
         vBufferMask64    __attribute__ ((aligned (16))),
        *vMsk64           __attribute__ ((aligned (16)));

void    *vMsk64unaligned = NULL;

void alignment_sse__clean() {if (vMsk64unaligned){ free(vMsk64unaligned); vMsk64unaligned = NULL;}}

#endif




unsigned int   prlength;




#ifdef __AVX512BW__

/**
 * AVX512BW alignment init read function : modify the read length when needed (but must not be changed too frequently).
 * @param readlength gives the read length (number of nucleotides inside the read)
 */

void alignment_avx512bw__setlength_quad(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+15) * 2;

  /* allocating/reallocating mask table */
  if (vMsk512unaligned)
    free(vMsk512unaligned);
  vMsk512unaligned = malloc(prlength * sizeof(VTYPE512) + 63);
  if (!vMsk512unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk512 = (void *) ((uintptr_t)(vMsk512unaligned + 63) & ~0x3f);

  /* init mask table */
  vMsk512[0] = EPI64_TYPE(_mm512_set)(0xff00000000000000LL,
                                      0x0000000000000000LL,
                                      0xff00000000000000LL,
                                      0x0000000000000000LL,
                                      0xff00000000000000LL,
                                      0x0000000000000000LL,
                                      0xff00000000000000LL,
                                      0x0000000000000000LL);
#ifdef DEBUG_SIMD
  {
    vector512_t Msk;
    Msk.v = vMsk512[0];
    fprintf(stderr,"[0]\t Msk:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(Msk.u64[7]),(Msk.u64[6]),(Msk.u64[5]),(Msk.u64[4]),(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk512[l] = vMsk512[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 15*2) {
        vMsk512[l] = EPI128_TYPE(_mm512_bsrli)(vMsk512[l],(1)*8);
      } else {
        /* mask at the beginning */
        if (l <= 15*2) {
          vMsk512[l] = SI512_TYPE(_mm512_or)(vMsk512[l-1],EPI128_TYPE(_mm512_bsrli)(vMsk512[l],(1)*8));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector512_t Msk;
      Msk.v = vMsk512[l];
      fprintf(stderr,"[0]\t Msk:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(Msk.u64[7]),(Msk.u64[6]),(Msk.u64[5]),(Msk.u64[4]),(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
    }
#endif
  }/* for l */
}


void alignment_avx512bw__setlength_octa(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+7) * 2;

  /* allocating/reallocating mask table */
  if (vMsk512unaligned)
    free(vMsk512unaligned);
  vMsk512unaligned = malloc(prlength * sizeof(VTYPE512) + 63);
  if (!vMsk512unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk512 = (void *) ((uintptr_t)(vMsk512unaligned + 63) & ~0x3f);

  /* init mask table */
  vMsk512[0] = EPI64_TYPE(_mm512_set)(0xff00000000000000LL,
                                      0xff00000000000000LL,
                                      0xff00000000000000LL,
                                      0xff00000000000000LL,
                                      0xff00000000000000LL,
                                      0xff00000000000000LL,
                                      0xff00000000000000LL,
                                      0xff00000000000000LL);
#ifdef DEBUG_SIMD
  {
    vector512_t Msk;
    Msk.v = vMsk512[0];
    fprintf(stderr,"[0]\t Msk:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(Msk.u64[7]),(Msk.u64[6]),(Msk.u64[5]),(Msk.u64[4]),(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk512[l] = vMsk512[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 7*2) {
        vMsk512[l] = EPI64_TYPE(_mm512_srli)(vMsk512[l],(1)*8);
      } else {
        /* mask at the beginning */
        if (l <= 7*2) {
          vMsk512[l] = SI512_TYPE(_mm512_or)(vMsk512[l-1],EPI64_TYPE(_mm512_srli)(vMsk512[l],(1)*8));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector512_t Msk;
      Msk.v = vMsk512[l];
      fprintf(stderr,"[0]\t Msk:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(Msk.u64[7]),(Msk.u64[6]),(Msk.u64[5]),(Msk.u64[4]),(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
    }
#endif
  }/* for l */
}


void alignment_avx512bw__setlength_hexa(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+3) * 2;

  /* allocating/reallocating mask table */
  if (vMsk512unaligned)
    free(vMsk512unaligned);
  vMsk512unaligned = malloc(prlength * sizeof(VTYPE512) + 63);
  if (!vMsk512unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk512 = (void *) ((uintptr_t)(vMsk512unaligned + 63) & ~0x3f);

  /* init mask table */
  vMsk512[0] = EPI32_TYPE(_mm512_set)(0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000);
#ifdef DEBUG_SIMD
  {
    vector512_t Msk;
    Msk.v = vMsk512[0];
    fprintf(stderr,"[0]\t Msk:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[15]),(Msk.u32[14]),(Msk.u32[13]),(Msk.u32[12]),(Msk.u32[11]),(Msk.u32[10]),(Msk.u32[9]),(Msk.u32[8]),(Msk.u32[7]),(Msk.u32[6]),(Msk.u32[5]),(Msk.u32[4]),(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk512[l] = vMsk512[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 3*2) {
        vMsk512[l] = EPI32_TYPE(_mm512_srli)(vMsk512[l],(1)*8);
      } else {
        /* mask at the beginning */
        if (l <= 3*2) {
          vMsk512[l] = SI512_TYPE(_mm512_or)(vMsk512[l-1],EPI32_TYPE(_mm512_srli)(vMsk512[l],(1)*8));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector512_t Msk;
      Msk.v = vMsk512[l];
      fprintf(stderr,"[0]\t Msk:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[15]),(Msk.u32[14]),(Msk.u32[13]),(Msk.u32[12]),(Msk.u32[11]),(Msk.u32[10]),(Msk.u32[9]),(Msk.u32[8]),(Msk.u32[7]),(Msk.u32[6]),(Msk.u32[5]),(Msk.u32[4]),(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
    }
#endif
  }/* for l */
}


void alignment_avx512bw__setlength_tria(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+1) * 2;

  /* allocating/reallocating mask table */
  if (vMsk512unaligned)
    free(vMsk512unaligned);
  vMsk512unaligned = malloc(prlength * sizeof(VTYPE512) + 63);
  if (!vMsk512unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk512 = (void *) ((uintptr_t)(vMsk512unaligned + 63) & ~0x3f);

  /* init mask table */
  vMsk512[0] = EPI32_TYPE(_mm512_set)(0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00,
                                      0xff00ff00);
#ifdef DEBUG_SIMD
  {
    vector512_t Msk;
    Msk.v = vMsk512[0];
    fprintf(stderr,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[31]),(Msk.u16[30]),(Msk.u16[29]),(Msk.u16[28]),(Msk.u16[27]),(Msk.u16[26]),(Msk.u16[25]),(Msk.u16[24]),(Msk.u16[23]),(Msk.u16[22]),(Msk.u16[21]),(Msk.u16[20]),(Msk.u16[19]),(Msk.u16[18]),(Msk.u16[17]),(Msk.u16[16]),(Msk.u16[15]),(Msk.u16[14]),(Msk.u16[13]),(Msk.u16[12]),(Msk.u16[11]),(Msk.u16[10]),(Msk.u16[9]),(Msk.u16[8]),(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk512[l] = vMsk512[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 1*2) {
        vMsk512[l] = EPI16_TYPE(_mm512_srli)(vMsk512[l],(1)*8);
      } else {
        /* mask at the beginning */
        if (l <= 1*2) {
          vMsk512[l] = SI512_TYPE(_mm512_or)(vMsk512[l-1],EPI32_TYPE(_mm512_srli)(vMsk512[l],(1)*8));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector512_t Msk;
      Msk.v = vMsk512[l];
      fprintf(stderr,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[31]),(Msk.u16[30]),(Msk.u16[29]),(Msk.u16[28]),(Msk.u16[27]),(Msk.u16[26]),(Msk.u16[25]),(Msk.u16[24]),(Msk.u16[23]),(Msk.u16[22]),(Msk.u16[21]),(Msk.u16[20]),(Msk.u16[19]),(Msk.u16[18]),(Msk.u16[17]),(Msk.u16[16]),(Msk.u16[15]),(Msk.u16[14]),(Msk.u16[13]),(Msk.u16[12]),(Msk.u16[11]),(Msk.u16[10]),(Msk.u16[9]),(Msk.u16[8]),(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
    }
#endif
  }/* for l */
}


/**
 * AVX512BW alignment init function : fix the scoring system and the length of the reads (must be called once before aligning)
 * @param match      inits the match score vector
 * @param mismatch   inits the mismatch penalty vector (positive value only)
 * @param gapopen    inits the gap penalty vector (positive value only)
 * @param gapextends inits the gap penalty vector (positive value only)
 * @param threshold  inits the scoring threshold (positive value only)
 * @param length     fixes the length of the reads that will be treated : this value can be changed
 * @see alignment_avx512bw__setlength_quad @see alignment_avx512bw__setlength_octa @see alignment_avx512bw__setlength_hexa @see alignment_avx512bw__setlength_tria
 *        (but must not be changed too frequently).
 */

void alignment_avx512bw__init_quad(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_avx512bw__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with AVX512BW instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes512          =  EPI64_TYPE(_mm512_set1)(1);
  vBufferMask512    =  EPI64_TYPE(_mm512_set)(0LL, 3LL, 0LL, 3LL, 0LL, 3LL, 0LL, 3LL);
  vThreshold512     =  EPI8_TYPE(_mm512_set1)(u_threshold);
  vIndelOpenS512    =  EPI8_TYPE(_mm512_set1)(gapopen);
  vIndelExtendsS512 =  EPI8_TYPE(_mm512_set1)(gapextends);
  vMatchS512        =  EPI8_TYPE(_mm512_set1)(match);
  vMismatchS512     =  EPI8_TYPE(_mm512_set1)(mismatch);

  alignment_avx512bw__setlength_quad(readlength);
}


void alignment_avx512bw__init_octa(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_avx512bw__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with AVX512BW instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes512          =  EPI64_TYPE(_mm512_set1)(1);
  vBufferMask512    =  EPI64_TYPE(_mm512_set)(3LL, 3LL, 3LL, 3LL, 3LL, 3LL, 3LL, 3LL);
  vThreshold512     =  EPI8_TYPE(_mm512_set1)(u_threshold);
  vIndelOpenS512    =  EPI8_TYPE(_mm512_set1)(gapopen);
  vIndelExtendsS512 =  EPI8_TYPE(_mm512_set1)(gapextends);
  vMatchS512        =  EPI8_TYPE(_mm512_set1)(match);
  vMismatchS512     =  EPI8_TYPE(_mm512_set1)(mismatch);

  alignment_avx512bw__setlength_octa(readlength);
}


void alignment_avx512bw__init_hexa(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_avx512bw__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with AVX512BW instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes512          =  EPI32_TYPE(_mm512_set1)(1);
  vBufferMask512    =  EPI32_TYPE(_mm512_set1)(3);
  vThreshold512     =  EPI8_TYPE(_mm512_set1)(u_threshold);
  vIndelOpenS512    =  EPI8_TYPE(_mm512_set1)(gapopen);
  vIndelExtendsS512 =  EPI8_TYPE(_mm512_set1)(gapextends);
  vMatchS512        =  EPI8_TYPE(_mm512_set1)(match);
  vMismatchS512     =  EPI8_TYPE(_mm512_set1)(mismatch);

  alignment_avx512bw__setlength_hexa(readlength);
}


void alignment_avx512bw__init_tria(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_avx512bw__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with AVX512BW instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes512          =  EPI16_TYPE(_mm512_set1)(1);
  vBufferMask512    =  EPI16_TYPE(_mm512_set1)(3);
  vThreshold512     =  EPI8_TYPE(_mm512_set1)(u_threshold);
  vIndelOpenS512    =  EPI8_TYPE(_mm512_set1)(gapopen);
  vIndelExtendsS512 =  EPI8_TYPE(_mm512_set1)(gapextends);
  vMatchS512        =  EPI8_TYPE(_mm512_set1)(match);
  vMismatchS512     =  EPI8_TYPE(_mm512_set1)(mismatch);

  alignment_avx512bw__setlength_tria(readlength);
}


/**
 * AVX512BW alignment align function : does a banded smith-waterman of the given read against two parts of the genome;
 * allows at most 1/2, 3/4, 7/8 or 15/16 indels on each side.
 * @param genome is the compressed genome (first nucleotide is the lower bit of the first byte)
 * @param pos_genome gives the list of positions (in term of nucleotides) to be aligned with the read
 *        (you must substract 1/2, 3/4, 7/8 or 15/16 potential indels according to the hit position)
 * @param read is the compressed read (first nucleotide is the lower bit of the first byte)
 * @return 0 if none is aligned to reach the given threshold (what should happened most of the time),
 *         or a bitmask 1<<(x) | 1<<(y) if x or y align correctly.
 */

unsigned int alignment_avx512bw__align_quad(unsigned char * genome,
                                            int * pos_genome,
                                            unsigned char * read) {
  VTYPE512 vA;
  VTYPE512 vB;

  VTYPE512 vMMax;
  VTYPE512 vM_old;
  VTYPE512 vM_old_old;
  VTYPE512 vI_old;

  unsigned char *             byte_pos_genome[4];
  unsigned int                 sub_pos_genome[4];
  {
    int d;
    for (d = 0; d < 4; d++) {
      byte_pos_genome[d] = genome + (pos_genome[d] >> 2);
       sub_pos_genome[d] = (pos_genome[d] & 3);
    }
  }

  vector512_t        vector_genome_buffer;
  vector512_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI512_TYPE(_mm512_setzero)();

  vector512_t        vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI512_TYPE(_mm512_setzero)();
  vM_old     = SI512_TYPE(_mm512_setzero)();
  vM_old_old = SI512_TYPE(_mm512_setzero)();
  vI_old     = SI512_TYPE(_mm512_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI512_TYPE(_mm512_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_QUAD512(sub_pos_genome,byte_pos_genome,
                        vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 16; d++) {
      VTYPE512 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_QUAD512(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = EPI128_TYPE(_mm512_bslli)(vB,(1));
      vB        = SI512_TYPE(_mm512_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE512 vLB;
        NEXTGENOSEQ_NOSUB_QUAD512(byte_pos_genome,
                                  vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = EPI128_TYPE(_mm512_bslli)(vB,(1));
        vB  = SI512_TYPE(_mm512_or)(vB,vLB);
      } else {
        VTYPE512 vLA;
        NEXTREADSEQ_QUAD512(read,
                            vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = EPI128_TYPE(_mm512_bslli)(vLA,(15));
        vA  = EPI128_TYPE(_mm512_bsrli)(vA,(1));
        vA  = SI512_TYPE(_mm512_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector512_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(A.u64[7]),(A.u64[6]),(A.u64[5]),(A.u64[4]),(A.u64[3]),(A.u64[2]),(A.u64[1]),(A.u64[0]));
        fprintf(stderr,"[1]\t   B:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(B.u64[7]),(B.u64[6]),(B.u64[5]),(B.u64[4]),(B.u64[3]),(B.u64[2]),(B.u64[1]),(B.u64[0]));
      }
#endif


      VTYPE512 vM;
      /* b) compute the matching score */
      {
        __mmask64 ab_MatchMask = EPI8_MASK_TYPE(_mm512_cmp)(vA,vB,_MM_CMPINT_EQ);
        VTYPE512  vM_add       = EPI8_TYPE(_mm512_maskz_mov)( ab_MatchMask,vMatchS512);
        VTYPE512  vM_sub       = EPI8_TYPE(_mm512_maskz_mov)(~ab_MatchMask,vMismatchS512);

#ifdef DEBUG_SIMD
        {
          vector512_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(S_a.u64[7]),(S_a.u64[6]),(S_a.u64[5]),(S_a.u64[4]),(S_a.u64[3]),(S_a.u64[2]),(S_a.u64[1]),(S_a.u64[0]));
          fprintf(stderr,"[1]\t S_s:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(S_s.u64[7]),(S_s.u64[6]),(S_a.u64[5]),(S_a.u64[4]),(S_s.u64[3]),(S_s.u64[2]),(S_a.u64[1]),(S_a.u64[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector512_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(M_old_old.u64[7]),(M_old_old.u64[6]),(M_old_old.u64[5]),(M_old_old.u64[4]),(M_old_old.u64[3]),(M_old_old.u64[2]),(M_old_old.u64[1]),(M_old_old.u64[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm512_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm512_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector512_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(    M.u64[7]),(    M.u64[6]),(    M.u64[5]),(    M.u64[4]),(    M.u64[3]),(    M.u64[2]),(    M.u64[1]),(    M.u64[0]));
        fprintf(stderr,"[1]\t  oM:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(M_old.u64[7]),(M_old.u64[6]),(M_old.u64[5]),(M_old.u64[4]),(M_old.u64[3]),(M_old.u64[2]),(M_old.u64[1]),(M_old.u64[0]));
        fprintf(stderr,"[1]\t  oI:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(I_old.u64[7]),(I_old.u64[6]),(I_old.u64[5]),(I_old.u64[4]),(I_old.u64[3]),(I_old.u64[2]),(I_old.u64[1]),(I_old.u64[0]));
      }
#endif

      VTYPE512 vI;
      {
        /* shift */
        VTYPE512 vM_old_shifted;
        VTYPE512 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = EPI128_TYPE(_mm512_bslli)(vM_old,(1));
          vI_old_shifted  = EPI128_TYPE(_mm512_bslli)(vI_old,(1));
        } else {
          vM_old_shifted  = EPI128_TYPE(_mm512_bsrli)(vM_old,(1));
          vI_old_shifted  = EPI128_TYPE(_mm512_bsrli)(vI_old,(1));
        }
        VTYPE512 vI_old_merge = EPU8_TYPE(_mm512_max)(vI_old,vI_old_shifted);
        VTYPE512 vM_old_merge = EPU8_TYPE(_mm512_max)(vM_old,vM_old_shifted);
        vI                    = EPU8_TYPE(_mm512_subs)(vI_old_merge,vIndelExtendsS512);
        VTYPE512 vIstart      = EPU8_TYPE(_mm512_subs)(vM_old_merge,vIndelOpenS512);
        vI                    = EPU8_TYPE(_mm512_max)(vI,vIstart);
        vM                    = EPU8_TYPE(_mm512_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector512_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(M.u64[7]),(M.u64[6]),(M.u64[5]),(M.u64[4]),(M.u64[3]),(M.u64[2]),(M.u64[1]),(M.u64[0]));
        fprintf(stderr,"[1]\t>  I:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(I.u64[7]),(I.u64[6]),(I.u64[5]),(I.u64[4]),(I.u64[3]),(I.u64[2]),(I.u64[1]),(I.u64[0]));
      }
#endif

      vM    = SI512_TYPE(_mm512_and)(vM,vMsk512[l]);
      vMMax = EPU8_TYPE(_mm512_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector512_t Msk,M,Max;
        Msk.v = vMsk512[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(Msk.u64[7]),(Msk.u64[6]),(Msk.u64[5]),(Msk.u64[4]),(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
        fprintf(stderr,"[1]\t>  M:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(  M.u64[7]),(  M.u64[6]),(  M.u64[5]),(  M.u64[4]),(  M.u64[3]),(  M.u64[2]),(  M.u64[1]),(  M.u64[0]));
        fprintf(stderr,"[1]\t>Max:%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx,%.16llx%.16llx\n",(Max.u64[7]),(Max.u64[6]),(Max.u64[5]),(Max.u64[4]),(Max.u64[3]),(Max.u64[2]),(Max.u64[1]),(Max.u64[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     --------------------------------,--------------------------------,--------------------------------,--------------------------------\n");
#endif
    } /* l */
  }
  {
    VTYPE512 vThresholdMask = EPU8_TYPE(_mm512_subs)(vMMax,vThreshold512);
    unsigned int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE512)/(2*(int)sizeof(uint64_t));x++) {
      uint64_t u0 =  ((vector512_t)vThresholdMask).u64[2*x];
      uint64_t u1 =  ((vector512_t)vThresholdMask).u64[2*x+1];
      if ((u0 != (uint64_t) 0) || (u1 != (uint64_t) 0)) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ================================,================================,================================,================================\n");
#endif
    return result;
  }
}


unsigned int alignment_avx512bw__align_octa(unsigned char * genome,
                                            int * pos_genome,
                                            unsigned char * read) {
  VTYPE512 vA;
  VTYPE512 vB;

  VTYPE512 vMMax;
  VTYPE512 vM_old;
  VTYPE512 vM_old_old;
  VTYPE512 vI_old;

  unsigned char *             byte_pos_genome[8];
  unsigned int                 sub_pos_genome[8];
  {
    int d;
    for (d = 0; d < 8; d++) {
      byte_pos_genome[d] = genome + (pos_genome[d] >> 2);
       sub_pos_genome[d] = (pos_genome[d] & 3);
    }
  }

  vector512_t        vector_genome_buffer;
  vector512_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI512_TYPE(_mm512_setzero)();

  vector512_t        vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI512_TYPE(_mm512_setzero)();
  vM_old     = SI512_TYPE(_mm512_setzero)();
  vM_old_old = SI512_TYPE(_mm512_setzero)();
  vI_old     = SI512_TYPE(_mm512_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI512_TYPE(_mm512_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_OCTA512(sub_pos_genome,byte_pos_genome,
                        vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 8; d++) {
      VTYPE512 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_OCTA512(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = EPI64_TYPE(_mm512_slli)(vB,(1)*8);
      vB        = SI512_TYPE(_mm512_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE512 vLB;
        NEXTGENOSEQ_NOSUB_OCTA512(byte_pos_genome,
                                  vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = EPI64_TYPE(_mm512_slli)(vB,(1)*8);
        vB  = SI512_TYPE(_mm512_or)(vB,vLB);
      } else {
        VTYPE512 vLA;
        NEXTREADSEQ_OCTA512(read,
                            vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = EPI64_TYPE(_mm512_slli)(vLA,(7)*8);
        vA  = EPI64_TYPE(_mm512_srli)(vA,(1)*8);
        vA  = SI512_TYPE(_mm512_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector512_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(A.u64[7]),(A.u64[6]),(A.u64[5]),(A.u64[4]),(A.u64[3]),(A.u64[2]),(A.u64[1]),(A.u64[0]));
        fprintf(stderr,"[1]\t   B:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(B.u64[7]),(B.u64[6]),(B.u64[5]),(B.u64[4]),(B.u64[3]),(B.u64[2]),(B.u64[1]),(B.u64[0]));
      }
#endif


      VTYPE512 vM;
      /* b) compute the matching score */
      {
        __mmask64 ab_MatchMask = EPI8_MASK_TYPE(_mm512_cmp)(vA,vB,_MM_CMPINT_EQ);
        VTYPE512  vM_add       = EPI8_TYPE(_mm512_maskz_mov)( ab_MatchMask,vMatchS512);
        VTYPE512  vM_sub       = EPI8_TYPE(_mm512_maskz_mov)(~ab_MatchMask,vMismatchS512);

#ifdef DEBUG_SIMD
        {
          vector512_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(S_a.u64[7]),(S_a.u64[6]),(S_a.u64[5]),(S_a.u64[4]),(S_a.u64[3]),(S_a.u64[2]),(S_a.u64[1]),(S_a.u64[0]));
          fprintf(stderr,"[1]\t S_s:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(S_s.u64[7]),(S_s.u64[6]),(S_s.u64[5]),(S_s.u64[4]),(S_s.u64[3]),(S_s.u64[2]),(S_s.u64[1]),(S_s.u64[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector512_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(M_old_old.u64[7]),(M_old_old.u64[6]),(M_old_old.u64[5]),(M_old_old.u64[4]),(M_old_old.u64[3]),(M_old_old.u64[2]),(M_old_old.u64[1]),(M_old_old.u64[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm512_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm512_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector512_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(    M.u64[7]),(    M.u64[6]),(    M.u64[5]),(    M.u64[4]),(    M.u64[3]),(    M.u64[2]),(    M.u64[1]),(    M.u64[0]));
        fprintf(stderr,"[1]\t  oM:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(M_old.u64[7]),(M_old.u64[6]),(M_old.u64[5]),(M_old.u64[4]),(M_old.u64[3]),(M_old.u64[2]),(M_old.u64[1]),(M_old.u64[0]));
        fprintf(stderr,"[1]\t  oI:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(I_old.u64[7]),(I_old.u64[6]),(I_old.u64[5]),(I_old.u64[4]),(I_old.u64[3]),(I_old.u64[2]),(I_old.u64[1]),(I_old.u64[0]));
      }
#endif

      VTYPE512 vI;
      {
        /* shift */
        VTYPE512 vM_old_shifted;
        VTYPE512 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = EPI64_TYPE(_mm512_slli)(vM_old,(1)*8);
          vI_old_shifted  = EPI64_TYPE(_mm512_slli)(vI_old,(1)*8);
        } else {
          vM_old_shifted  = EPI64_TYPE(_mm512_srli)(vM_old,(1)*8);
          vI_old_shifted  = EPI64_TYPE(_mm512_srli)(vI_old,(1)*8);
        }
        VTYPE512 vI_old_merge = EPU8_TYPE(_mm512_max)(vI_old,vI_old_shifted);
        VTYPE512 vM_old_merge = EPU8_TYPE(_mm512_max)(vM_old,vM_old_shifted);
        vI                    = EPU8_TYPE(_mm512_subs)(vI_old_merge,vIndelExtendsS512);
        VTYPE512 vIstart      = EPU8_TYPE(_mm512_subs)(vM_old_merge,vIndelOpenS512);
        vI                    = EPU8_TYPE(_mm512_max)(vI,vIstart);
        vM                    = EPU8_TYPE(_mm512_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector512_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(M.u64[7]),(M.u64[6]),(M.u64[5]),(M.u64[4]),(M.u64[3]),(M.u64[2]),(M.u64[1]),(M.u64[0]));
        fprintf(stderr,"[1]\t>  I:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(I.u64[7]),(I.u64[6]),(I.u64[5]),(I.u64[4]),(I.u64[3]),(I.u64[2]),(I.u64[1]),(I.u64[0]));
      }
#endif

      vM    = SI512_TYPE(_mm512_and)(vM,vMsk512[l]);
      vMMax = EPU8_TYPE(_mm512_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector512_t Msk,M,Max;
        Msk.v = vMsk512[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(Msk.u64[7]),(Msk.u64[6]),(Msk.u64[5]),(Msk.u64[4]),(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
        fprintf(stderr,"[1]\t>  M:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(  M.u64[7]),(  M.u64[6]),(  M.u64[5]),(  M.u64[4]),(  M.u64[3]),(  M.u64[2]),(  M.u64[1]),(  M.u64[0]));
        fprintf(stderr,"[1]\t>Max:%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx,%.16llx\n",(Max.u64[7]),(Max.u64[6]),(Max.u64[5]),(Max.u64[4]),(Max.u64[3]),(Max.u64[2]),(Max.u64[1]),(Max.u64[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     ----------------,----------------,----------------,----------------,----------------,----------------,----------------,----------------\n");
#endif
    } /* l */
  }
  {
    VTYPE512 vThresholdMask = EPU8_TYPE(_mm512_subs)(vMMax,vThreshold512);
    unsigned long long int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE512)/(int)sizeof(uint64_t);x++) {
      uint64_t u =  ((vector512_t)vThresholdMask).u64[x];
      if (u != (uint64_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ================,================,================,================,================,================,================,================\n");
#endif
    return result;
  }
}


unsigned int alignment_avx512bw__align_hexa(unsigned char * genome,
                                            int * pos_genome,
                                            unsigned char * read) {
  VTYPE512 vA;
  VTYPE512 vB;

  VTYPE512 vMMax;
  VTYPE512 vM_old;
  VTYPE512 vM_old_old;
  VTYPE512 vI_old;

  unsigned char *             byte_pos_genome[16];
  unsigned int                 sub_pos_genome[16];
  {
    int d;
    for (d = 0; d < 16; d++) {
      byte_pos_genome[d] = genome + (pos_genome[d] >> 2);
       sub_pos_genome[d] = (pos_genome[d] & 3);
    }
  }

  vector512_t        vector_genome_buffer;
  vector512_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI512_TYPE(_mm512_setzero)();

  vector512_t        vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI512_TYPE(_mm512_setzero)();
  vM_old     = SI512_TYPE(_mm512_setzero)();
  vM_old_old = SI512_TYPE(_mm512_setzero)();
  vI_old     = SI512_TYPE(_mm512_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI512_TYPE(_mm512_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_HEXA512(sub_pos_genome,byte_pos_genome,
                        vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 4; d++) {
      VTYPE512 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_HEXA512(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = EPI32_TYPE(_mm512_slli)(vB,(1)*8);
      vB        = SI512_TYPE(_mm512_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE512 vLB;
        NEXTGENOSEQ_NOSUB_HEXA512(byte_pos_genome,
                                  vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = EPI32_TYPE(_mm512_slli)(vB,(1)*8);
        vB  = SI512_TYPE(_mm512_or)(vB,vLB);
      } else {
        VTYPE512 vLA;
        NEXTREADSEQ_HEXA512(read,
                            vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = EPI32_TYPE(_mm512_slli)(vLA,(3)*8);
        vA  = EPI32_TYPE(_mm512_srli)(vA,(1)*8);
        vA  = SI512_TYPE(_mm512_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector512_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(A.u32[15]),(A.u32[14]),(A.u32[13]),(A.u32[12]),(A.u32[11]),(A.u32[10]),(A.u32[9]),(A.u32[8]),(A.u32[7]),(A.u32[6]),(A.u32[5]),(A.u32[4]),(A.u32[3]),(A.u32[2]),(A.u32[1]),(A.u32[0]));
        fprintf(stderr,"[1]\t   B:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(B.u32[15]),(B.u32[14]),(B.u32[13]),(B.u32[12]),(B.u32[11]),(B.u32[10]),(B.u32[9]),(B.u32[8]),(B.u32[7]),(B.u32[6]),(B.u32[5]),(B.u32[4]),(B.u32[3]),(B.u32[2]),(B.u32[1]),(B.u32[0]));
      }
#endif


      VTYPE512 vM;
      /* b) compute the matching score */
      {
        __mmask64 ab_MatchMask = EPI8_MASK_TYPE(_mm512_cmp)(vA,vB,_MM_CMPINT_EQ);
        VTYPE512  vM_add       = EPI8_TYPE(_mm512_maskz_mov)( ab_MatchMask,vMatchS512);
        VTYPE512  vM_sub       = EPI8_TYPE(_mm512_maskz_mov)(~ab_MatchMask,vMismatchS512);

#ifdef DEBUG_SIMD
        {
          vector512_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(S_a.u32[15]),(S_a.u32[14]),(S_a.u32[13]),(S_a.u32[12]),(S_a.u32[11]),(S_a.u32[10]),(S_a.u32[9]),(S_a.u32[8]),(S_a.u32[7]),(S_a.u32[6]),(S_a.u32[5]),(S_a.u32[4]),(S_a.u32[3]),(S_a.u32[2]),(S_a.u32[1]),(S_a.u32[0]));
          fprintf(stderr,"[1]\t S_s:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(S_s.u32[15]),(S_s.u32[14]),(S_s.u32[13]),(S_s.u32[12]),(S_s.u32[11]),(S_s.u32[10]),(S_s.u32[9]),(S_s.u32[8]),(S_s.u32[7]),(S_s.u32[6]),(S_s.u32[5]),(S_s.u32[4]),(S_s.u32[3]),(S_s.u32[2]),(S_s.u32[1]),(S_s.u32[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector512_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(M_old_old.u32[15]),(M_old_old.u32[14]),(M_old_old.u32[13]),(M_old_old.u32[12]),(M_old_old.u32[11]),(M_old_old.u32[10]),(M_old_old.u32[9]),(M_old_old.u32[8]),(M_old_old.u32[7]),(M_old_old.u32[6]),(M_old_old.u32[5]),(M_old_old.u32[4]),(M_old_old.u32[3]),(M_old_old.u32[2]),(M_old_old.u32[1]),(M_old_old.u32[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm512_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm512_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector512_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(    M.u32[15]),(    M.u32[14]),(    M.u32[13]),(    M.u32[12]),(    M.u32[11]),(    M.u32[10]),(    M.u32[9]),(    M.u32[8]),(    M.u32[7]),(    M.u32[6]),(    M.u32[5]),(    M.u32[4]),(    M.u32[3]),(    M.u32[2]),(    M.u32[1]),(    M.u32[0]));
        fprintf(stderr,"[1]\t  oM:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(M_old.u32[15]),(M_old.u32[14]),(M_old.u32[13]),(M_old.u32[12]),(M_old.u32[11]),(M_old.u32[10]),(M_old.u32[9]),(M_old.u32[8]),(M_old.u32[7]),(M_old.u32[6]),(M_old.u32[5]),(M_old.u32[4]),(M_old.u32[3]),(M_old.u32[2]),(M_old.u32[1]),(M_old.u32[0]));
        fprintf(stderr,"[1]\t  oI:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(I_old.u32[15]),(I_old.u32[14]),(I_old.u32[13]),(I_old.u32[12]),(I_old.u32[11]),(I_old.u32[10]),(I_old.u32[9]),(I_old.u32[8]),(I_old.u32[7]),(I_old.u32[6]),(I_old.u32[5]),(I_old.u32[4]),(I_old.u32[3]),(I_old.u32[2]),(I_old.u32[1]),(I_old.u32[0]));
      }
#endif

      VTYPE512 vI;
      {
        /* shift */
        VTYPE512 vM_old_shifted;
        VTYPE512 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = EPI32_TYPE(_mm512_slli)(vM_old,(1)*8);
          vI_old_shifted  = EPI32_TYPE(_mm512_slli)(vI_old,(1)*8);
        } else {
          vM_old_shifted  = EPI32_TYPE(_mm512_srli)(vM_old,(1)*8);
          vI_old_shifted  = EPI32_TYPE(_mm512_srli)(vI_old,(1)*8);
        }
        VTYPE512 vI_old_merge = EPU8_TYPE(_mm512_max)(vI_old,vI_old_shifted);
        VTYPE512 vM_old_merge = EPU8_TYPE(_mm512_max)(vM_old,vM_old_shifted);
        vI                    = EPU8_TYPE(_mm512_subs)(vI_old_merge,vIndelExtendsS512);
        VTYPE512 vIstart      = EPU8_TYPE(_mm512_subs)(vM_old_merge,vIndelOpenS512);
        vI                    = EPU8_TYPE(_mm512_max)(vI,vIstart);
        vM                    = EPU8_TYPE(_mm512_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector512_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(M.u32[15]),(M.u32[14]),(M.u32[13]),(M.u32[12]),(M.u32[11]),(M.u32[10]),(M.u32[9]),(M.u32[8]),(M.u32[7]),(M.u32[6]),(M.u32[5]),(M.u32[4]),(M.u32[3]),(M.u32[2]),(M.u32[1]),(M.u32[0]));
        fprintf(stderr,"[1]\t>  I:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(I.u32[15]),(I.u32[14]),(I.u32[13]),(I.u32[12]),(I.u32[11]),(I.u32[10]),(I.u32[9]),(I.u32[8]),(I.u32[7]),(I.u32[6]),(I.u32[5]),(I.u32[4]),(I.u32[3]),(I.u32[2]),(I.u32[1]),(I.u32[0]));
      }
#endif

      vM    = SI512_TYPE(_mm512_and)(vM,vMsk512[l]);
      vMMax = EPU8_TYPE(_mm512_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector512_t Msk,M,Max;
        Msk.v = vMsk512[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[15]),(Msk.u32[14]),(Msk.u32[13]),(Msk.u32[12]),(Msk.u32[11]),(Msk.u32[10]),(Msk.u32[9]),(Msk.u32[8]),(Msk.u32[7]),(Msk.u32[6]),(Msk.u32[5]),(Msk.u32[4]),(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
        fprintf(stderr,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(  M.u32[15]),(  M.u32[14]),(  M.u32[13]),(  M.u32[12]),(  M.u32[11]),(  M.u32[10]),(  M.u32[9]),(  M.u32[8]),(  M.u32[7]),(  M.u32[6]),(  M.u32[5]),(  M.u32[4]),(  M.u32[3]),(  M.u32[2]),(  M.u32[1]),(  M.u32[0]));
        fprintf(stderr,"[1]\t>Max:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(Max.u32[15]),(Max.u32[14]),(Max.u32[13]),(Max.u32[12]),(Max.u32[11]),(Max.u32[10]),(Max.u32[9]),(Max.u32[8]),(Max.u32[7]),(Max.u32[6]),(Max.u32[5]),(Max.u32[4]),(Max.u32[3]),(Max.u32[2]),(Max.u32[1]),(Max.u32[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     --------,--------,--------,--------,--------,--------,--------,--------,--------,--------,--------,--------,--------,--------,--------,--------\n");
#endif
    } /* l */
  }
  {
    VTYPE512 vThresholdMask = EPU8_TYPE(_mm512_subs)(vMMax,vThreshold512);
    unsigned long long int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE512)/(int)sizeof(uint32_t);x++) {
      uint32_t u =  ((vector512_t)vThresholdMask).u32[x];
      if (u != (uint32_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ========,========,========,========,========,========,========,========,========,========,========,========,========,========,========,========\n");
#endif
    return result;
  }
}


unsigned int alignment_avx512bw__align_tria(unsigned char * genome,
                                            int * pos_genome,
                                            unsigned char * read) {
  VTYPE512 vA;
  VTYPE512 vB;

  VTYPE512 vMMax;
  VTYPE512 vM_old;
  VTYPE512 vM_old_old;
  VTYPE512 vI_old;

  unsigned char *             byte_pos_genome[32];
  unsigned int                 sub_pos_genome[32];
  {
    int d;
    for (d = 0; d < 32; d++) {
      byte_pos_genome[d] = genome + (pos_genome[d] >> 2);
       sub_pos_genome[d] = (pos_genome[d] & 3);
    }
  }

  vector512_t        vector_genome_buffer;
  vector512_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI512_TYPE(_mm512_setzero)();

  vector512_t        vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI512_TYPE(_mm512_setzero)();
  vM_old     = SI512_TYPE(_mm512_setzero)();
  vM_old_old = SI512_TYPE(_mm512_setzero)();
  vI_old     = SI512_TYPE(_mm512_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI512_TYPE(_mm512_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_TRIA512(sub_pos_genome,byte_pos_genome,
                        vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 2; d++) {
      VTYPE512 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_TRIA512(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = EPI16_TYPE(_mm512_slli)(vB,(1)*8);
      vB        = SI512_TYPE(_mm512_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE512 vLB;
        NEXTGENOSEQ_NOSUB_TRIA512(byte_pos_genome,
                                  vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = EPI16_TYPE(_mm512_slli)(vB,(1)*8);
        vB  = SI512_TYPE(_mm512_or)(vB,vLB);
      } else {
        VTYPE512 vLA;
        NEXTREADSEQ_TRIA512(read,
                            vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = EPI16_TYPE(_mm512_slli)(vLA,(1)*8);
        vA  = EPI16_TYPE(_mm512_srli)(vA,(1)*8);
        vA  = SI512_TYPE(_mm512_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector512_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(A.u16[31]),(A.u16[30]),(A.u16[29]),(A.u16[28]),(A.u16[27]),(A.u16[26]),(A.u16[25]),(A.u16[24]),(A.u16[23]),(A.u16[22]),(A.u16[21]),(A.u16[20]),(A.u16[19]),(A.u16[18]),(A.u16[17]),(A.u16[16]),(A.u16[15]),(A.u16[14]),(A.u16[13]),(A.u16[12]),(A.u16[11]),(A.u16[10]),(A.u16[9]),(A.u16[8]),(A.u16[7]),(A.u16[6]),(A.u16[5]),(A.u16[4]),(A.u16[3]),(A.u16[2]),(A.u16[1]),(A.u16[0]));
        fprintf(stderr,"[1]\t   B:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(B.u16[31]),(B.u16[30]),(B.u16[29]),(B.u16[28]),(B.u16[27]),(B.u16[26]),(B.u16[25]),(B.u16[24]),(B.u16[23]),(B.u16[22]),(B.u16[21]),(B.u16[20]),(B.u16[19]),(B.u16[18]),(B.u16[17]),(B.u16[16]),(B.u16[15]),(B.u16[14]),(B.u16[13]),(B.u16[12]),(B.u16[11]),(B.u16[10]),(B.u16[9]),(B.u16[8]),(B.u16[7]),(B.u16[6]),(B.u16[5]),(B.u16[4]),(B.u16[3]),(B.u16[2]),(B.u16[1]),(B.u16[0]));
      }
#endif


      VTYPE512 vM;
      /* b) compute the matching score */
      {
        __mmask64 ab_MatchMask = EPI8_MASK_TYPE(_mm512_cmp)(vA,vB,_MM_CMPINT_EQ);
        VTYPE512  vM_add       = EPI8_TYPE(_mm512_maskz_mov)( ab_MatchMask,vMatchS512);
        VTYPE512  vM_sub       = EPI8_TYPE(_mm512_maskz_mov)(~ab_MatchMask,vMismatchS512);

#ifdef DEBUG_SIMD
        {
          vector512_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(S_a.u16[31]),(S_a.u16[30]),(S_a.u16[29]),(S_a.u16[28]),(S_a.u16[27]),(S_a.u16[26]),(S_a.u16[25]),(S_a.u16[24]),(S_a.u16[23]),(S_a.u16[22]),(S_a.u16[21]),(S_a.u16[20]),(S_a.u16[19]),(S_a.u16[18]),(S_a.u16[17]),(S_a.u16[16]),(S_a.u16[15]),(S_a.u16[14]),(S_a.u16[13]),(S_a.u16[12]),(S_a.u16[11]),(S_a.u16[10]),(S_a.u16[9]),(S_a.u16[8]),(S_a.u16[7]),(S_a.u16[6]),(S_a.u16[5]),(S_a.u16[4]),(S_a.u16[3]),(S_a.u16[2]),(S_a.u16[1]),(S_a.u16[0]));
          fprintf(stderr,"[1]\t S_s:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(S_s.u16[31]),(S_s.u16[30]),(S_s.u16[29]),(S_s.u16[28]),(S_s.u16[27]),(S_s.u16[26]),(S_s.u16[25]),(S_s.u16[24]),(S_s.u16[23]),(S_s.u16[22]),(S_s.u16[21]),(S_s.u16[20]),(S_s.u16[19]),(S_s.u16[18]),(S_s.u16[17]),(S_s.u16[16]),(S_s.u16[15]),(S_s.u16[14]),(S_s.u16[13]),(S_s.u16[12]),(S_s.u16[11]),(S_s.u16[10]),(S_s.u16[9]),(S_s.u16[8]),(S_s.u16[7]),(S_s.u16[6]),(S_s.u16[5]),(S_s.u16[4]),(S_s.u16[3]),(S_s.u16[2]),(S_s.u16[1]),(S_s.u16[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector512_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M_old_old.u16[31]),(M_old_old.u16[30]),(M_old_old.u16[29]),(M_old_old.u16[28]),(M_old_old.u16[27]),(M_old_old.u16[26]),(M_old_old.u16[25]),(M_old_old.u16[24]),(M_old_old.u16[23]),(M_old_old.u16[22]),(M_old_old.u16[21]),(M_old_old.u16[20]),(M_old_old.u16[19]),(M_old_old.u16[18]),(M_old_old.u16[17]),(M_old_old.u16[16]),(M_old_old.u16[15]),(M_old_old.u16[14]),(M_old_old.u16[13]),(M_old_old.u16[12]),(M_old_old.u16[11]),(M_old_old.u16[10]),(M_old_old.u16[9]),(M_old_old.u16[8]),(M_old_old.u16[7]),(M_old_old.u16[6]),(M_old_old.u16[5]),(M_old_old.u16[4]),(M_old_old.u16[3]),(M_old_old.u16[2]),(M_old_old.u16[1]),(M_old_old.u16[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm512_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm512_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector512_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(    M.u16[31]),(    M.u16[30]),(    M.u16[29]),(    M.u16[28]),(    M.u16[27]),(    M.u16[26]),(    M.u16[25]),(    M.u16[24]),(    M.u16[23]),(    M.u16[22]),(    M.u16[21]),(    M.u16[20]),(    M.u16[19]),(    M.u16[18]),(    M.u16[17]),(    M.u16[16]),(    M.u16[15]),(    M.u16[14]),(    M.u16[13]),(    M.u16[12]),(    M.u16[11]),(    M.u16[10]),(    M.u16[9]),(    M.u16[8]),(    M.u16[7]),(    M.u16[6]),(    M.u16[5]),(    M.u16[4]),(    M.u16[3]),(    M.u16[2]),(    M.u16[1]),(    M.u16[0]));
        fprintf(stderr,"[1]\t  oM:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M_old.u16[31]),(M_old.u16[30]),(M_old.u16[29]),(M_old.u16[28]),(M_old.u16[27]),(M_old.u16[26]),(M_old.u16[25]),(M_old.u16[24]),(M_old.u16[23]),(M_old.u16[22]),(M_old.u16[21]),(M_old.u16[20]),(M_old.u16[19]),(M_old.u16[18]),(M_old.u16[17]),(M_old.u16[16]),(M_old.u16[15]),(M_old.u16[14]),(M_old.u16[13]),(M_old.u16[12]),(M_old.u16[11]),(M_old.u16[10]),(M_old.u16[9]),(M_old.u16[8]),(M_old.u16[7]),(M_old.u16[6]),(M_old.u16[5]),(M_old.u16[4]),(M_old.u16[3]),(M_old.u16[2]),(M_old.u16[1]),(M_old.u16[0]));
        fprintf(stderr,"[1]\t  oI:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(I_old.u16[31]),(I_old.u16[30]),(I_old.u16[29]),(I_old.u16[28]),(I_old.u16[27]),(I_old.u16[26]),(I_old.u16[25]),(I_old.u16[24]),(I_old.u16[23]),(I_old.u16[22]),(I_old.u16[21]),(I_old.u16[20]),(I_old.u16[19]),(I_old.u16[18]),(I_old.u16[17]),(I_old.u16[16]),(I_old.u16[15]),(I_old.u16[14]),(I_old.u16[13]),(I_old.u16[12]),(I_old.u16[11]),(I_old.u16[10]),(I_old.u16[9]),(I_old.u16[8]),(I_old.u16[7]),(I_old.u16[6]),(I_old.u16[5]),(I_old.u16[4]),(I_old.u16[3]),(I_old.u16[2]),(I_old.u16[1]),(I_old.u16[0]));
      }
#endif

      VTYPE512 vI;
      {
        /* shift */
        VTYPE512 vM_old_shifted;
        VTYPE512 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = EPI16_TYPE(_mm512_slli)(vM_old,(1)*8);
          vI_old_shifted  = EPI16_TYPE(_mm512_slli)(vI_old,(1)*8);
        } else {
          vM_old_shifted  = EPI16_TYPE(_mm512_srli)(vM_old,(1)*8);
          vI_old_shifted  = EPI16_TYPE(_mm512_srli)(vI_old,(1)*8);
        }
        VTYPE512 vI_old_merge = EPU8_TYPE(_mm512_max)(vI_old,vI_old_shifted);
        VTYPE512 vM_old_merge = EPU8_TYPE(_mm512_max)(vM_old,vM_old_shifted);
        vI                    = EPU8_TYPE(_mm512_subs)(vI_old_merge,vIndelExtendsS512);
        VTYPE512 vIstart      = EPU8_TYPE(_mm512_subs)(vM_old_merge,vIndelOpenS512);
        vI                    = EPU8_TYPE(_mm512_max)(vI,vIstart);
        vM                    = EPU8_TYPE(_mm512_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector512_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M.u16[31]),(M.u16[30]),(M.u16[29]),(M.u16[28]),(M.u16[27]),(M.u16[26]),(M.u16[25]),(M.u16[24]),(M.u16[23]),(M.u16[22]),(M.u16[21]),(M.u16[20]),(M.u16[19]),(M.u16[18]),(M.u16[17]),(M.u16[16]),(M.u16[15]),(M.u16[14]),(M.u16[13]),(M.u16[12]),(M.u16[11]),(M.u16[10]),(M.u16[9]),(M.u16[8]),(M.u16[7]),(M.u16[6]),(M.u16[5]),(M.u16[4]),(M.u16[3]),(M.u16[2]),(M.u16[1]),(M.u16[0]));
        fprintf(stderr,"[1]\t>  I:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(I.u16[31]),(I.u16[30]),(I.u16[29]),(I.u16[28]),(I.u16[27]),(I.u16[26]),(I.u16[25]),(I.u16[24]),(I.u16[23]),(I.u16[22]),(I.u16[21]),(I.u16[20]),(I.u16[19]),(I.u16[18]),(I.u16[17]),(I.u16[16]),(I.u16[15]),(I.u16[14]),(I.u16[13]),(I.u16[12]),(I.u16[11]),(I.u16[10]),(I.u16[9]),(I.u16[8]),(I.u16[7]),(I.u16[6]),(I.u16[5]),(I.u16[4]),(I.u16[3]),(I.u16[2]),(I.u16[1]),(I.u16[0]));
      }
#endif

      vM    = SI512_TYPE(_mm512_and)(vM,vMsk512[l]);
      vMMax = EPU8_TYPE(_mm512_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector512_t Msk,M,Max;
        Msk.v = vMsk512[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[31]),(Msk.u16[30]),(Msk.u16[29]),(Msk.u16[28]),(Msk.u16[27]),(Msk.u16[26]),(Msk.u16[25]),(Msk.u16[24]),(Msk.u16[23]),(Msk.u16[22]),(Msk.u16[21]),(Msk.u16[20]),(Msk.u16[19]),(Msk.u16[18]),(Msk.u16[17]),(Msk.u16[16]),(Msk.u16[15]),(Msk.u16[14]),(Msk.u16[13]),(Msk.u16[12]),(Msk.u16[11]),(Msk.u16[10]),(Msk.u16[9]),(Msk.u16[8]),(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
        fprintf(stderr,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(  M.u16[31]),(  M.u16[30]),(  M.u16[29]),(  M.u16[28]),(  M.u16[27]),(  M.u16[26]),(  M.u16[25]),(  M.u16[24]),(  M.u16[23]),(  M.u16[22]),(  M.u16[21]),(  M.u16[20]),(  M.u16[19]),(  M.u16[18]),(  M.u16[17]),(  M.u16[16]),(  M.u16[15]),(  M.u16[14]),(  M.u16[13]),(  M.u16[12]),(  M.u16[11]),(  M.u16[10]),(  M.u16[9]),(  M.u16[8]),(  M.u16[7]),(  M.u16[6]),(  M.u16[5]),(  M.u16[4]),(  M.u16[3]),(  M.u16[2]),(  M.u16[1]),(  M.u16[0]));
        fprintf(stderr,"[1]\t>Max:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Max.u16[31]),(Max.u16[30]),(Max.u16[29]),(Max.u16[28]),(Max.u16[27]),(Max.u16[26]),(Max.u16[25]),(Max.u16[24]),(Max.u16[23]),(Max.u16[22]),(Max.u16[21]),(Max.u16[20]),(Max.u16[19]),(Max.u16[18]),(Max.u16[17]),(Max.u16[16]),(Max.u16[15]),(Max.u16[14]),(Max.u16[13]),(Max.u16[12]),(Max.u16[11]),(Max.u16[10]),(Max.u16[9]),(Max.u16[8]),(Max.u16[7]),(Max.u16[6]),(Max.u16[5]),(Max.u16[4]),(Max.u16[3]),(Max.u16[2]),(Max.u16[1]),(Max.u16[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     ----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----\n");
#endif
    } /* l */
  }
  {
    VTYPE512 vThresholdMask = EPU8_TYPE(_mm512_subs)(vMMax,vThreshold512);
    unsigned long long int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE512)/(int)sizeof(uint16_t);x++) {
      uint16_t u =  ((vector512_t)vThresholdMask).u16[x];
      if (u != (uint16_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====\n");
#endif
    return result;
  }
}

#endif




#ifdef __AVX2__

/**
 * AVX2 alignment init read function : modify the read length when needed (but must not be changed too frequently).
 * @param readlength gives the read length (number of nucleotides inside the read)
 */

void alignment_avx2__setlength_pair(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+15) * 2;

  /* allocating/reallocating mask table */
  if (vMsk256unaligned)
    free(vMsk256unaligned);
  vMsk256unaligned = malloc(prlength * sizeof(VTYPE256) + 31);
  if (!vMsk256unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk256 = (void *) ((uintptr_t)(vMsk256unaligned + 31) & ~0x1f);

  /* init mask table */
  vMsk256[0] = EPI64X_TYPE(_mm256_set)(0xff00000000000000LL,
                                       0x0000000000000000LL,
                                       0xff00000000000000LL,
                                       0x0000000000000000LL);
#ifdef DEBUG_SIMD
  {
    vector256_t Msk;
    Msk.v = vMsk256[0];
    fprintf(stderr,"[0]\t Msk:%.16llx%.16llx,%.16llx%.16llx\n",(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk256[l] = vMsk256[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 15*2) {
        vMsk256[l] = SI256_TYPE(_mm256_srli)(vMsk256[l],(1));
      } else {
        /* mask at the beginning */
        if (l <= 15*2) {
          vMsk256[l] = SI256_TYPE(_mm256_or)(vMsk256[l-1],SI256_TYPE(_mm256_srli)(vMsk256[l],(1)));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector256_t Msk;
      Msk.v = vMsk256[l];
      fprintf(stderr,"[0]\t Msk:%.16llx%.16llx,%.16llx%.16llx\n",(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
    }
#endif
  }/* for l */
}


void alignment_avx2__setlength_quad(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+7) * 2;

  /* allocating/reallocating mask table */
  if (vMsk256unaligned)
    free(vMsk256unaligned);
  vMsk256unaligned = malloc(prlength * sizeof(VTYPE256) + 31);
  if (!vMsk256unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk256 = (void *) ((uintptr_t)(vMsk256unaligned + 31) & ~0x1f);

  /* init mask table */
  vMsk256[0] = EPI64X_TYPE(_mm256_set)(0xff00000000000000LL,
                                       0xff00000000000000LL,
                                       0xff00000000000000LL,
                                       0xff00000000000000LL);
#ifdef DEBUG_SIMD
  {
    vector256_t Msk;
    Msk.v = vMsk256[0];
    fprintf(stderr,"[0]\t Msk:%.16llx,%.16llx,%.16llx,%.16llx\n",(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk256[l] = vMsk256[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 7*2) {
        vMsk256[l] = EPI64_TYPE(_mm256_srli)(vMsk256[l],(1)*8);
      } else {
        /* mask at the beginning */
        if (l <= 7*2) {
          vMsk256[l] = SI256_TYPE(_mm256_or)(vMsk256[l-1],EPI64_TYPE(_mm256_srli)(vMsk256[l],(1)*8));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector256_t Msk;
      Msk.v = vMsk256[l];
      fprintf(stderr,"[0]\t Msk:%.16llx,%.16llx,%.16llx,%.16llx\n",(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
    }
#endif
  }/* for l */
}


void alignment_avx2__setlength_octa(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+3) * 2;

  /* allocating/reallocating mask table */
  if (vMsk256unaligned)
    free(vMsk256unaligned);
  vMsk256unaligned = malloc(prlength * sizeof(VTYPE256) + 31);
  if (!vMsk256unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk256 = (void *) ((uintptr_t)(vMsk256unaligned + 31) & ~0x1f);

  /* init mask table */
  vMsk256[0] = EPI32_TYPE(_mm256_set)(0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000,
                                      0xff000000);
#ifdef DEBUG_SIMD
  {
    vector256_t Msk;
    Msk.v = vMsk256[0];
    fprintf(stderr,"[0]\t Msk:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[7]),(Msk.u32[6]),(Msk.u32[5]),(Msk.u32[4]),(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk256[l] = vMsk256[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 3*2) {
        vMsk256[l] = EPI32_TYPE(_mm256_srli)(vMsk256[l],(1)*8);
      } else {
        /* mask at the beginning */
        if (l <= 3*2) {
          vMsk256[l] = SI256_TYPE(_mm256_or)(vMsk256[l-1],EPI32_TYPE(_mm256_srli)(vMsk256[l],(1)*8));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector256_t Msk;
      Msk.v = vMsk256[l];
      fprintf(stderr,"[0]\t Msk:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[7]),(Msk.u32[6]),(Msk.u32[5]),(Msk.u32[4]),(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
    }
#endif
  }/* for l */
}


void alignment_avx2__setlength_hexa(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+1) * 2;

  /* allocating/reallocating mask table */
  if (vMsk256unaligned)
    free(vMsk256unaligned);
  vMsk256unaligned = malloc(prlength * sizeof(VTYPE256) + 31);
  if (!vMsk256unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk256 = (void *) ((uintptr_t)(vMsk256unaligned + 31) & ~0x1f);

  /* init mask table */
  vMsk256[0] = EPI16_TYPE(_mm256_set)(0xff00,0xff00,
                                      0xff00,0xff00,
                                      0xff00,0xff00,
                                      0xff00,0xff00,
                                      0xff00,0xff00,
                                      0xff00,0xff00,
                                      0xff00,0xff00,
                                      0xff00,0xff00);
#ifdef DEBUG_SIMD
  {
    vector256_t Msk;
    Msk.v = vMsk256[0];
    fprintf(stderr,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[15]),(Msk.u16[14]),(Msk.u16[13]),(Msk.u16[12]),(Msk.u16[11]),(Msk.u16[10]),(Msk.u16[9]),(Msk.u16[8]),(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk256[l] = vMsk256[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 1*2) {
        vMsk256[l] = EPI16_TYPE(_mm256_srli)(vMsk256[l],(1)*8);
      } else {
        /* mask at the beginning */
        if (l <= 1*2) {
          vMsk256[l] = SI256_TYPE(_mm256_or)(vMsk256[l-1],EPI32_TYPE(_mm256_srli)(vMsk256[l],(1)*8));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector256_t Msk;
      Msk.v = vMsk256[l];
      fprintf(stderr,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[15]),(Msk.u16[14]),(Msk.u16[13]),(Msk.u16[12]),(Msk.u16[11]),(Msk.u16[10]),(Msk.u16[9]),(Msk.u16[8]),(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
    }
#endif
  }/* for l */
}


/**
 * AVX2 alignment init function : fix the scoring system and the length of the reads (must be called once before aligning)
 * @param match      inits the match score vector
 * @param mismatch   inits the mismatch penalty vector (positive value only)
 * @param gapopen    inits the gap penalty vector (positive value only)
 * @param gapextends inits the gap penalty vector (positive value only)
 * @param threshold  inits the scoring threshold (positive value only)
 * @param length     fixes the length of the reads that will be treated : this value can be changed
 * @see alignment_avx2__setlength_pair @see alignment_avx2__setlength_quad @see alignment_avx2__setlength_octa @see alignment_avx2__setlength_hexa
 *        (but must not be changed too frequently).
 */

void alignment_avx2__init_pair(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_avx2__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with AVX2 instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes256          =  EPI64X_TYPE(_mm256_set1)(1);
  vBufferMask256    =  EPI64X_TYPE(_mm256_set)(0LL, 3LL, 0LL, 3LL);
  vThreshold256     =  EPI8_TYPE(_mm256_set1)(u_threshold);
  vIndelOpenS256    =  EPI8_TYPE(_mm256_set1)(gapopen);
  vIndelExtendsS256 =  EPI8_TYPE(_mm256_set1)(gapextends);
  vMatchS256        =  EPI8_TYPE(_mm256_set1)(match);
  vMismatchS256     =  EPI8_TYPE(_mm256_set1)(mismatch);

  alignment_avx2__setlength_pair(readlength);
}


void alignment_avx2__init_quad(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_avx2__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with AVX2 instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes256          =  EPI64X_TYPE(_mm256_set1)(1);
  vBufferMask256    =  EPI64X_TYPE(_mm256_set)(3LL, 3LL, 3LL, 3LL);
  vThreshold256     =  EPI8_TYPE(_mm256_set1)(u_threshold);
  vIndelOpenS256    =  EPI8_TYPE(_mm256_set1)(gapopen);
  vIndelExtendsS256 =  EPI8_TYPE(_mm256_set1)(gapextends);
  vMatchS256        =  EPI8_TYPE(_mm256_set1)(match);
  vMismatchS256     =  EPI8_TYPE(_mm256_set1)(mismatch);

  alignment_avx2__setlength_quad(readlength);
}


void alignment_avx2__init_octa(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_avx2__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with AVX2 instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes256          =  EPI32_TYPE(_mm256_set1)(1);
  vBufferMask256    =  EPI32_TYPE(_mm256_set1)(3);
  vThreshold256     =  EPI8_TYPE(_mm256_set1)(u_threshold);
  vIndelOpenS256    =  EPI8_TYPE(_mm256_set1)(gapopen);
  vIndelExtendsS256 =  EPI8_TYPE(_mm256_set1)(gapextends);
  vMatchS256        =  EPI8_TYPE(_mm256_set1)(match);
  vMismatchS256     =  EPI8_TYPE(_mm256_set1)(mismatch);

  alignment_avx2__setlength_octa(readlength);
}


void alignment_avx2__init_hexa(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_avx2__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with AVX2 instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes256          =  EPI16_TYPE(_mm256_set1)(1);
  vBufferMask256    =  EPI16_TYPE(_mm256_set1)(3);
  vThreshold256     =  EPI8_TYPE(_mm256_set1)(u_threshold);
  vIndelOpenS256    =  EPI8_TYPE(_mm256_set1)(gapopen);
  vIndelExtendsS256 =  EPI8_TYPE(_mm256_set1)(gapextends);
  vMatchS256        =  EPI8_TYPE(_mm256_set1)(match);
  vMismatchS256     =  EPI8_TYPE(_mm256_set1)(mismatch);

  alignment_avx2__setlength_hexa(readlength);
}


/**
 * AVX2 alignment align function : does a banded smith-waterman of the given read against two parts of the genome;
 * allows at most 1/2, 3/4, 7/8 or 15/16 indels on each side.
 * @param genome is the compressed genome (first nucleotide is the lower bit of the first byte)
 * @param pos_genome gives the list of positions (in term of nucleotides) to be aligned with the read
 *        (you must substract 1/2, 3/4, 7/8 or 15/16 potential indels according to the hit position)
 * @param read is the compressed read (first nucleotide is the lower bit of the first byte)
 * @return 0 if none is aligned to reach the given threshold (what should happened most of the time),
 *         or a bitmask 1<<(x) | 1<<(y) if x or y align correctly.
 */

unsigned int alignment_avx2__align_pair(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read) {
  VTYPE256 vA;
  VTYPE256 vB;

  VTYPE256 vMMax;
  VTYPE256 vM_old;
  VTYPE256 vM_old_old;
  VTYPE256 vI_old;

  unsigned char *             byte_pos_genome[2];
  unsigned int                 sub_pos_genome[2];
  {
    int d;
    for (d = 0; d < 2; d++) {
      byte_pos_genome[d] = genome + (pos_genome[d] >> 2);
       sub_pos_genome[d] = (pos_genome[d] & 3);
    }
  }

  vector256_t        vector_genome_buffer;
  vector256_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI256_TYPE(_mm256_setzero)();

  vector256_t        vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI256_TYPE(_mm256_setzero)();
  vM_old     = SI256_TYPE(_mm256_setzero)();
  vM_old_old = SI256_TYPE(_mm256_setzero)();
  vI_old     = SI256_TYPE(_mm256_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI256_TYPE(_mm256_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_PAIR256(sub_pos_genome,byte_pos_genome,
                        vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 16; d++) {
      VTYPE256 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_PAIR256(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = SI256_TYPE(_mm256_slli)(vB,(1));
      vB        = SI256_TYPE(_mm256_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE256 vLB;
        NEXTGENOSEQ_NOSUB_PAIR256(byte_pos_genome,
                                  vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = SI256_TYPE(_mm256_slli)(vB,(1));
        vB  = SI256_TYPE(_mm256_or)(vB,vLB);
      } else {
        VTYPE256 vLA;
        NEXTREADSEQ_PAIR256(read,
                            vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = SI256_TYPE(_mm256_slli)(vLA,(15));
        vA  = SI256_TYPE(_mm256_srli)(vA,(1));
        vA  = SI256_TYPE(_mm256_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector256_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.16llx%.16llx,%.16llx%.16llx\n",(A.u64[3]),(A.u64[2]),(A.u64[1]),(A.u64[0]));
        fprintf(stderr,"[1]\t   B:%.16llx%.16llx,%.16llx%.16llx\n",(B.u64[3]),(B.u64[2]),(B.u64[1]),(B.u64[0]));
      }
#endif


      VTYPE256 vM;
      /* b) compute the matching score */
      {
        VTYPE256 vM_ab_MatchMask = EPI8_TYPE(_mm256_cmpeq)(vA,vB);
        VTYPE256 vM_add = SI256_TYPE(_mm256_and)(vM_ab_MatchMask,vMatchS256);
        VTYPE256 vM_sub = SI256_TYPE(_mm256_andnot)(vM_ab_MatchMask,vMismatchS256);

#ifdef DEBUG_SIMD
        {
          vector256_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.16llx%.16llx,%.16llx%.16llx\n",(S_a.u64[3]),(S_a.u64[2]),(S_a.u64[1]),(S_a.u64[0]));
          fprintf(stderr,"[1]\t S_s:%.16llx%.16llx,%.16llx%.16llx\n",(S_s.u64[3]),(S_s.u64[2]),(S_a.u64[1]),(S_a.u64[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector256_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.16llx%.16llx,%.16llx%.16llx\n",(M_old_old.u64[3]),(M_old_old.u64[2]),(M_old_old.u64[1]),(M_old_old.u64[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm256_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm256_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector256_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.16llx%.16llx,%.16llx%.16llx\n",(    M.u64[3]),(    M.u64[2]),(    M.u64[1]),(    M.u64[0]));
        fprintf(stderr,"[1]\t  oM:%.16llx%.16llx,%.16llx%.16llx\n",(M_old.u64[3]),(M_old.u64[2]),(M_old.u64[1]),(M_old.u64[0]));
        fprintf(stderr,"[1]\t  oI:%.16llx%.16llx,%.16llx%.16llx\n",(I_old.u64[3]),(I_old.u64[2]),(I_old.u64[1]),(I_old.u64[0]));
      }
#endif

      VTYPE256 vI;
      {
        /* shift */
        VTYPE256 vM_old_shifted;
        VTYPE256 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = SI256_TYPE(_mm256_slli)(vM_old,(1));
          vI_old_shifted  = SI256_TYPE(_mm256_slli)(vI_old,(1));
        } else {
          vM_old_shifted  = SI256_TYPE(_mm256_srli)(vM_old,(1));
          vI_old_shifted  = SI256_TYPE(_mm256_srli)(vI_old,(1));
        }
        VTYPE256 vI_old_merge = EPU8_TYPE(_mm256_max)(vI_old,vI_old_shifted);
        VTYPE256 vM_old_merge = EPU8_TYPE(_mm256_max)(vM_old,vM_old_shifted);
        vI                    = EPU8_TYPE(_mm256_subs)(vI_old_merge,vIndelExtendsS256);
        VTYPE256 vIstart      = EPU8_TYPE(_mm256_subs)(vM_old_merge,vIndelOpenS256);
        vI                    = EPU8_TYPE(_mm256_max)(vI,vIstart);
        vM                    = EPU8_TYPE(_mm256_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector256_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.16llx%.16llx,%.16llx%.16llx\n",(M.u64[3]),(M.u64[2]),(M.u64[1]),(M.u64[0]));
        fprintf(stderr,"[1]\t>  I:%.16llx%.16llx,%.16llx%.16llx\n",(I.u64[3]),(I.u64[2]),(I.u64[1]),(I.u64[0]));
      }
#endif

      vM    = SI256_TYPE(_mm256_and)(vM,vMsk256[l]);
      vMMax = EPU8_TYPE(_mm256_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector256_t Msk,M,Max;
        Msk.v = vMsk256[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.16llx%.16llx,%.16llx%.16llx\n",(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
        fprintf(stderr,"[1]\t>  M:%.16llx%.16llx,%.16llx%.16llx\n",(  M.u64[3]),(  M.u64[2]),(  M.u64[1]),(  M.u64[0]));
        fprintf(stderr,"[1]\t>Max:%.16llx%.16llx,%.16llx%.16llx\n",(Max.u64[3]),(Max.u64[2]),(Max.u64[1]),(Max.u64[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     --------------------------------,--------------------------------\n");
#endif
    } /* l */
  }
  {
    VTYPE256 vThresholdMask = EPU8_TYPE(_mm256_subs)(vMMax,vThreshold256);
    unsigned long long int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE256)/(2*(int)sizeof(uint64_t));x++) {
      uint64_t u0 =  ((vector256_t)vThresholdMask).u64[2*x];
      uint64_t u1 =  ((vector256_t)vThresholdMask).u64[2*x+1];
      if ((u0 != (uint64_t) 0) || (u1 != (uint64_t) 0)) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ================================,================================\n");
#endif
    return result;
  }
}


unsigned int alignment_avx2__align_quad(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read) {
  VTYPE256 vA;
  VTYPE256 vB;

  VTYPE256 vMMax;
  VTYPE256 vM_old;
  VTYPE256 vM_old_old;
  VTYPE256 vI_old;

  unsigned char *             byte_pos_genome[4];
  unsigned int                 sub_pos_genome[4];
  {
    int d;
    for (d = 0; d < 4; d++) {
      byte_pos_genome[d] = genome + (pos_genome[d] >> 2);
       sub_pos_genome[d] = (pos_genome[d] & 3);
    }
  }

  vector256_t        vector_genome_buffer;
  vector256_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI256_TYPE(_mm256_setzero)();

  vector256_t        vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI256_TYPE(_mm256_setzero)();
  vM_old     = SI256_TYPE(_mm256_setzero)();
  vM_old_old = SI256_TYPE(_mm256_setzero)();
  vI_old     = SI256_TYPE(_mm256_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI256_TYPE(_mm256_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_QUAD256(sub_pos_genome,byte_pos_genome,
                        vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 8; d++) {
      VTYPE256 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_QUAD256(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = EPI64_TYPE(_mm256_slli)(vB,(1)*8);
      vB        = SI256_TYPE(_mm256_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE256 vLB;
        NEXTGENOSEQ_NOSUB_QUAD256(byte_pos_genome,
                                  vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = EPI64_TYPE(_mm256_slli)(vB,(1)*8);
        vB  = SI256_TYPE(_mm256_or)(vB,vLB);
      } else {
        VTYPE256 vLA;
        NEXTREADSEQ_QUAD256(read,
                            vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = EPI64_TYPE(_mm256_slli)(vLA,(7)*8);
        vA  = EPI64_TYPE(_mm256_srli)(vA,(1)*8);
        vA  = SI256_TYPE(_mm256_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector256_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.16llx,%.16llx,%.16llx,%.16llx\n",(A.u64[3]),(A.u64[2]),(A.u64[1]),(A.u64[0]));
        fprintf(stderr,"[1]\t   B:%.16llx,%.16llx,%.16llx,%.16llx\n",(B.u64[3]),(B.u64[2]),(B.u64[1]),(B.u64[0]));
      }
#endif


      VTYPE256 vM;
      /* b) compute the matching score */
      {
        VTYPE256 vM_ab_MatchMask = EPI8_TYPE(_mm256_cmpeq)(vA,vB);
        VTYPE256 vM_add = SI256_TYPE(_mm256_and)(vM_ab_MatchMask,vMatchS256);
        VTYPE256 vM_sub = SI256_TYPE(_mm256_andnot)(vM_ab_MatchMask,vMismatchS256);

#ifdef DEBUG_SIMD
        {
          vector256_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.16llx,%.16llx,%.16llx,%.16llx\n",(S_a.u64[3]),(S_a.u64[2]),(S_a.u64[1]),(S_a.u64[0]));
          fprintf(stderr,"[1]\t S_s:%.16llx,%.16llx,%.16llx,%.16llx\n",(S_s.u64[3]),(S_s.u64[2]),(S_s.u64[1]),(S_s.u64[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector256_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.16llx,%.16llx,%.16llx,%.16llx\n",(M_old_old.u64[3]),(M_old_old.u64[2]),(M_old_old.u64[1]),(M_old_old.u64[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm256_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm256_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector256_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.16llx,%.16llx,%.16llx,%.16llx\n",(    M.u64[3]),(    M.u64[2]),(    M.u64[1]),(    M.u64[0]));
        fprintf(stderr,"[1]\t  oM:%.16llx,%.16llx,%.16llx,%.16llx\n",(M_old.u64[3]),(M_old.u64[2]),(M_old.u64[1]),(M_old.u64[0]));
        fprintf(stderr,"[1]\t  oI:%.16llx,%.16llx,%.16llx,%.16llx\n",(I_old.u64[3]),(I_old.u64[2]),(I_old.u64[1]),(I_old.u64[0]));
      }
#endif

      VTYPE256 vI;
      {
        /* shift */
        VTYPE256 vM_old_shifted;
        VTYPE256 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = EPI64_TYPE(_mm256_slli)(vM_old,(1)*8);
          vI_old_shifted  = EPI64_TYPE(_mm256_slli)(vI_old,(1)*8);
        } else {
          vM_old_shifted  = EPI64_TYPE(_mm256_srli)(vM_old,(1)*8);
          vI_old_shifted  = EPI64_TYPE(_mm256_srli)(vI_old,(1)*8);
        }
        VTYPE256 vI_old_merge = EPU8_TYPE(_mm256_max)(vI_old,vI_old_shifted);
        VTYPE256 vM_old_merge = EPU8_TYPE(_mm256_max)(vM_old,vM_old_shifted);
        vI                    = EPU8_TYPE(_mm256_subs)(vI_old_merge,vIndelExtendsS256);
        VTYPE256 vIstart      = EPU8_TYPE(_mm256_subs)(vM_old_merge,vIndelOpenS256);
        vI                    = EPU8_TYPE(_mm256_max)(vI,vIstart);
        vM                    = EPU8_TYPE(_mm256_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector256_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.16llx,%.16llx,%.16llx,%.16llx\n",(M.u64[3]),(M.u64[2]),(M.u64[1]),(M.u64[0]));
        fprintf(stderr,"[1]\t>  I:%.16llx,%.16llx,%.16llx,%.16llx\n",(I.u64[3]),(I.u64[2]),(I.u64[1]),(I.u64[0]));
      }
#endif

      vM    = SI256_TYPE(_mm256_and)(vM,vMsk256[l]);
      vMMax = EPU8_TYPE(_mm256_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector256_t Msk,M,Max;
        Msk.v = vMsk256[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.16llx,%.16llx,%.16llx,%.16llx\n",(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
        fprintf(stderr,"[1]\t>  M:%.16llx,%.16llx,%.16llx,%.16llx\n",(  M.u64[3]),(  M.u64[2]),(  M.u64[1]),(  M.u64[0]));
        fprintf(stderr,"[1]\t>Max:%.16llx,%.16llx,%.16llx,%.16llx\n",(Max.u64[3]),(Max.u64[2]),(Max.u64[1]),(Max.u64[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     ----------------,----------------,----------------,----------------\n");
#endif
    } /* l */
  }
  {
    VTYPE256 vThresholdMask = EPU8_TYPE(_mm256_subs)(vMMax,vThreshold256);
    unsigned long long int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE256)/(int)sizeof(uint64_t);x++) {
      uint64_t u =  ((vector256_t)vThresholdMask).u64[x];
      if (u != (uint64_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ================,================,================,================\n");
#endif
    return result;
  }
}


unsigned int alignment_avx2__align_octa(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read) {
  VTYPE256 vA;
  VTYPE256 vB;

  VTYPE256 vMMax;
  VTYPE256 vM_old;
  VTYPE256 vM_old_old;
  VTYPE256 vI_old;

  unsigned char *             byte_pos_genome[8];
  unsigned int                 sub_pos_genome[8];
  {
    int d;
    for (d = 0; d < 8; d++) {
      byte_pos_genome[d] = genome + (pos_genome[d] >> 2);
       sub_pos_genome[d] = (pos_genome[d] & 3);
    }
  }

  vector256_t        vector_genome_buffer;
  vector256_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI256_TYPE(_mm256_setzero)();

  vector256_t        vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI256_TYPE(_mm256_setzero)();
  vM_old     = SI256_TYPE(_mm256_setzero)();
  vM_old_old = SI256_TYPE(_mm256_setzero)();
  vI_old     = SI256_TYPE(_mm256_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI256_TYPE(_mm256_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_OCTA256(sub_pos_genome,byte_pos_genome,
                        vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 4; d++) {
      VTYPE256 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_OCTA256(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = EPI32_TYPE(_mm256_slli)(vB,(1)*8);
      vB        = SI256_TYPE(_mm256_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE256 vLB;
        NEXTGENOSEQ_NOSUB_OCTA256(byte_pos_genome,
                                  vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = EPI32_TYPE(_mm256_slli)(vB,(1)*8);
        vB  = SI256_TYPE(_mm256_or)(vB,vLB);
      } else {
        VTYPE256 vLA;
        NEXTREADSEQ_OCTA256(read,
                            vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = EPI32_TYPE(_mm256_slli)(vLA,(3)*8);
        vA  = EPI32_TYPE(_mm256_srli)(vA,(1)*8);
        vA  = SI256_TYPE(_mm256_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector256_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(A.u32[7]),(A.u32[6]),(A.u32[5]),(A.u32[4]),(A.u32[3]),(A.u32[2]),(A.u32[1]),(A.u32[0]));
        fprintf(stderr,"[1]\t   B:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(B.u32[7]),(B.u32[6]),(B.u32[5]),(B.u32[4]),(B.u32[3]),(B.u32[2]),(B.u32[1]),(B.u32[0]));
      }
#endif


      VTYPE256 vM;
      /* b) compute the matching score */
      {
        VTYPE256 vM_ab_MatchMask = EPI8_TYPE(_mm256_cmpeq)(vA,vB);
        VTYPE256 vM_add = SI256_TYPE(_mm256_and)(vM_ab_MatchMask,vMatchS256);
        VTYPE256 vM_sub = SI256_TYPE(_mm256_andnot)(vM_ab_MatchMask,vMismatchS256);

#ifdef DEBUG_SIMD
        {
          vector256_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(S_a.u32[7]),(S_a.u32[6]),(S_a.u32[5]),(S_a.u32[4]),(S_a.u32[3]),(S_a.u32[2]),(S_a.u32[1]),(S_a.u32[0]));
          fprintf(stderr,"[1]\t S_s:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(S_s.u32[7]),(S_s.u32[6]),(S_s.u32[5]),(S_s.u32[4]),(S_s.u32[3]),(S_s.u32[2]),(S_s.u32[1]),(S_s.u32[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector256_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(M_old_old.u32[7]),(M_old_old.u32[6]),(M_old_old.u32[5]),(M_old_old.u32[4]),(M_old_old.u32[3]),(M_old_old.u32[2]),(M_old_old.u32[1]),(M_old_old.u32[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm256_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm256_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector256_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(    M.u32[7]),(    M.u32[6]),(    M.u32[5]),(    M.u32[4]),(    M.u32[3]),(    M.u32[2]),(    M.u32[1]),(    M.u32[0]));
        fprintf(stderr,"[1]\t  oM:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(M_old.u32[7]),(M_old.u32[6]),(M_old.u32[5]),(M_old.u32[4]),(M_old.u32[3]),(M_old.u32[2]),(M_old.u32[1]),(M_old.u32[0]));
        fprintf(stderr,"[1]\t  oI:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(I_old.u32[7]),(I_old.u32[6]),(I_old.u32[5]),(I_old.u32[4]),(I_old.u32[3]),(I_old.u32[2]),(I_old.u32[1]),(I_old.u32[0]));
      }
#endif

      VTYPE256 vI;
      {
        /* shift */
        VTYPE256 vM_old_shifted;
        VTYPE256 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = EPI32_TYPE(_mm256_slli)(vM_old,(1)*8);
          vI_old_shifted  = EPI32_TYPE(_mm256_slli)(vI_old,(1)*8);
        } else {
          vM_old_shifted  = EPI32_TYPE(_mm256_srli)(vM_old,(1)*8);
          vI_old_shifted  = EPI32_TYPE(_mm256_srli)(vI_old,(1)*8);
        }
        VTYPE256 vI_old_merge = EPU8_TYPE(_mm256_max)(vI_old,vI_old_shifted);
        VTYPE256 vM_old_merge = EPU8_TYPE(_mm256_max)(vM_old,vM_old_shifted);
        vI                    = EPU8_TYPE(_mm256_subs)(vI_old_merge,vIndelExtendsS256);
        VTYPE256 vIstart      = EPU8_TYPE(_mm256_subs)(vM_old_merge,vIndelOpenS256);
        vI                    = EPU8_TYPE(_mm256_max)(vI,vIstart);
        vM                    = EPU8_TYPE(_mm256_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector256_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(M.u32[7]),(M.u32[6]),(M.u32[5]),(M.u32[4]),(M.u32[3]),(M.u32[2]),(M.u32[1]),(M.u32[0]));
        fprintf(stderr,"[1]\t>  I:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(I.u32[7]),(I.u32[6]),(I.u32[5]),(I.u32[4]),(I.u32[3]),(I.u32[2]),(I.u32[1]),(I.u32[0]));
      }
#endif

      vM    = SI256_TYPE(_mm256_and)(vM,vMsk256[l]);
      vMMax = EPU8_TYPE(_mm256_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector256_t Msk,M,Max;
        Msk.v = vMsk256[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[7]),(Msk.u32[6]),(Msk.u32[5]),(Msk.u32[4]),(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
        fprintf(stderr,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(  M.u32[7]),(  M.u32[6]),(  M.u32[5]),(  M.u32[4]),(  M.u32[3]),(  M.u32[2]),(  M.u32[1]),(  M.u32[0]));
        fprintf(stderr,"[1]\t>Max:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(Max.u32[7]),(Max.u32[6]),(Max.u32[5]),(Max.u32[4]),(Max.u32[3]),(Max.u32[2]),(Max.u32[1]),(Max.u32[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     --------,--------,--------,--------,--------,--------,--------,--------\n");
#endif
    } /* l */
  }
  {
    VTYPE256 vThresholdMask = EPU8_TYPE(_mm256_subs)(vMMax,vThreshold256);
    unsigned long long int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE256)/(int)sizeof(uint32_t);x++) {
      uint32_t u =  ((vector256_t)vThresholdMask).u32[x];
      if (u != (uint32_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ========,========,========,========,========,========,========,========\n");
#endif
    return result;
  }
}


unsigned int alignment_avx2__align_hexa(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read) {
  VTYPE256 vA;
  VTYPE256 vB;

  VTYPE256 vMMax;
  VTYPE256 vM_old;
  VTYPE256 vM_old_old;
  VTYPE256 vI_old;

  unsigned char *             byte_pos_genome[16];
  unsigned int                 sub_pos_genome[16];
  {
    int d;
    for (d = 0; d < 16; d++) {
      byte_pos_genome[d] = genome + (pos_genome[d] >> 2);
       sub_pos_genome[d] = (pos_genome[d] & 3);
    }
  }

  vector256_t        vector_genome_buffer;
  vector256_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI256_TYPE(_mm256_setzero)();

  vector256_t        vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI256_TYPE(_mm256_setzero)();
  vM_old     = SI256_TYPE(_mm256_setzero)();
  vM_old_old = SI256_TYPE(_mm256_setzero)();
  vI_old     = SI256_TYPE(_mm256_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI256_TYPE(_mm256_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_HEXA256(sub_pos_genome,byte_pos_genome,
                        vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 2; d++) {
      VTYPE256 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_HEXA256(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = EPI16_TYPE(_mm256_slli)(vB,(1)*8);
      vB        = SI256_TYPE(_mm256_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE256 vLB;
        NEXTGENOSEQ_NOSUB_HEXA256(byte_pos_genome,
                                  vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = EPI16_TYPE(_mm256_slli)(vB,(1)*8);
        vB  = SI256_TYPE(_mm256_or)(vB,vLB);
      } else {
        VTYPE256 vLA;
        NEXTREADSEQ_HEXA256(read,
                            vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = EPI16_TYPE(_mm256_slli)(vLA,(1)*8);
        vA  = EPI16_TYPE(_mm256_srli)(vA,(1)*8);
        vA  = SI256_TYPE(_mm256_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector256_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(A.u16[15]),(A.u16[14]),(A.u16[13]),(A.u16[12]),(A.u16[11]),(A.u16[10]),(A.u16[9]),(A.u16[8]),(A.u16[7]),(A.u16[6]),(A.u16[5]),(A.u16[4]),(A.u16[3]),(A.u16[2]),(A.u16[1]),(A.u16[0]));
        fprintf(stderr,"[1]\t   B:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(B.u16[15]),(B.u16[14]),(B.u16[13]),(B.u16[12]),(B.u16[11]),(B.u16[10]),(B.u16[9]),(B.u16[8]),(B.u16[7]),(B.u16[6]),(B.u16[5]),(B.u16[4]),(B.u16[3]),(B.u16[2]),(B.u16[1]),(B.u16[0]));
      }
#endif


      VTYPE256 vM;
      /* b) compute the matching score */
      {
        VTYPE256 vM_ab_MatchMask = EPI8_TYPE(_mm256_cmpeq)(vA,vB);
        VTYPE256 vM_add = SI256_TYPE(_mm256_and)(vM_ab_MatchMask,vMatchS256);
        VTYPE256 vM_sub = SI256_TYPE(_mm256_andnot)(vM_ab_MatchMask,vMismatchS256);

#ifdef DEBUG_SIMD
        {
          vector256_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(S_a.u16[15]),(S_a.u16[14]),(S_a.u16[13]),(S_a.u16[12]),(S_a.u16[11]),(S_a.u16[10]),(S_a.u16[9]),(S_a.u16[8]),(S_a.u16[7]),(S_a.u16[6]),(S_a.u16[5]),(S_a.u16[4]),(S_a.u16[3]),(S_a.u16[2]),(S_a.u16[1]),(S_a.u16[0]));
          fprintf(stderr,"[1]\t S_s:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(S_s.u16[15]),(S_s.u16[14]),(S_s.u16[13]),(S_s.u16[12]),(S_s.u16[11]),(S_s.u16[10]),(S_s.u16[9]),(S_s.u16[8]),(S_s.u16[7]),(S_s.u16[6]),(S_s.u16[5]),(S_s.u16[4]),(S_s.u16[3]),(S_s.u16[2]),(S_s.u16[1]),(S_s.u16[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector256_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M_old_old.u16[15]),(M_old_old.u16[14]),(M_old_old.u16[13]),(M_old_old.u16[12]),(M_old_old.u16[11]),(M_old_old.u16[10]),(M_old_old.u16[9]),(M_old_old.u16[8]),(M_old_old.u16[7]),(M_old_old.u16[6]),(M_old_old.u16[5]),(M_old_old.u16[4]),(M_old_old.u16[3]),(M_old_old.u16[2]),(M_old_old.u16[1]),(M_old_old.u16[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm256_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm256_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector256_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(    M.u16[15]),(    M.u16[14]),(    M.u16[13]),(    M.u16[12]),(    M.u16[11]),(    M.u16[10]),(    M.u16[9]),(    M.u16[8]),(    M.u16[7]),(    M.u16[6]),(    M.u16[5]),(    M.u16[4]),(    M.u16[3]),(    M.u16[2]),(    M.u16[1]),(    M.u16[0]));
        fprintf(stderr,"[1]\t  oM:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M_old.u16[15]),(M_old.u16[14]),(M_old.u16[13]),(M_old.u16[12]),(M_old.u16[11]),(M_old.u16[10]),(M_old.u16[9]),(M_old.u16[8]),(M_old.u16[7]),(M_old.u16[6]),(M_old.u16[5]),(M_old.u16[4]),(M_old.u16[3]),(M_old.u16[2]),(M_old.u16[1]),(M_old.u16[0]));
        fprintf(stderr,"[1]\t  oI:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(I_old.u16[15]),(I_old.u16[14]),(I_old.u16[13]),(I_old.u16[12]),(I_old.u16[11]),(I_old.u16[10]),(I_old.u16[9]),(I_old.u16[8]),(I_old.u16[7]),(I_old.u16[6]),(I_old.u16[5]),(I_old.u16[4]),(I_old.u16[3]),(I_old.u16[2]),(I_old.u16[1]),(I_old.u16[0]));
      }
#endif

      VTYPE256 vI;
      {
        /* shift */
        VTYPE256 vM_old_shifted;
        VTYPE256 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = EPI16_TYPE(_mm256_slli)(vM_old,(1)*8);
          vI_old_shifted  = EPI16_TYPE(_mm256_slli)(vI_old,(1)*8);
        } else {
          vM_old_shifted  = EPI16_TYPE(_mm256_srli)(vM_old,(1)*8);
          vI_old_shifted  = EPI16_TYPE(_mm256_srli)(vI_old,(1)*8);
        }
        VTYPE256 vI_old_merge = EPU8_TYPE(_mm256_max)(vI_old,vI_old_shifted);
        VTYPE256 vM_old_merge = EPU8_TYPE(_mm256_max)(vM_old,vM_old_shifted);
        vI                    = EPU8_TYPE(_mm256_subs)(vI_old_merge,vIndelExtendsS256);
        VTYPE256 vIstart      = EPU8_TYPE(_mm256_subs)(vM_old_merge,vIndelOpenS256);
        vI                    = EPU8_TYPE(_mm256_max)(vI,vIstart);
        vM                    = EPU8_TYPE(_mm256_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector256_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M.u16[15]),(M.u16[14]),(M.u16[13]),(M.u16[12]),(M.u16[11]),(M.u16[10]),(M.u16[9]),(M.u16[8]),(M.u16[7]),(M.u16[6]),(M.u16[5]),(M.u16[4]),(M.u16[3]),(M.u16[2]),(M.u16[1]),(M.u16[0]));
        fprintf(stderr,"[1]\t>  I:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(I.u16[15]),(I.u16[14]),(I.u16[13]),(I.u16[12]),(I.u16[11]),(I.u16[10]),(I.u16[9]),(I.u16[8]),(I.u16[7]),(I.u16[6]),(I.u16[5]),(I.u16[4]),(I.u16[3]),(I.u16[2]),(I.u16[1]),(I.u16[0]));
      }
#endif

      vM    = SI256_TYPE(_mm256_and)(vM,vMsk256[l]);
      vMMax = EPU8_TYPE(_mm256_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector256_t Msk,M,Max;
        Msk.v = vMsk256[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[15]),(Msk.u16[14]),(Msk.u16[13]),(Msk.u16[12]),(Msk.u16[11]),(Msk.u16[10]),(Msk.u16[9]),(Msk.u16[8]),(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
        fprintf(stderr,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(  M.u16[15]),(  M.u16[14]),(  M.u16[13]),(  M.u16[12]),(  M.u16[11]),(  M.u16[10]),(  M.u16[9]),(  M.u16[8]),(  M.u16[7]),(  M.u16[6]),(  M.u16[5]),(  M.u16[4]),(  M.u16[3]),(  M.u16[2]),(  M.u16[1]),(  M.u16[0]));
        fprintf(stderr,"[1]\t>Max:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Max.u16[15]),(Max.u16[14]),(Max.u16[13]),(Max.u16[12]),(Max.u16[11]),(Max.u16[10]),(Max.u16[9]),(Max.u16[8]),(Max.u16[7]),(Max.u16[6]),(Max.u16[5]),(Max.u16[4]),(Max.u16[3]),(Max.u16[2]),(Max.u16[1]),(Max.u16[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     ----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----\n");
#endif
    } /* l */
  }
  {
    VTYPE256 vThresholdMask = EPU8_TYPE(_mm256_subs)(vMMax,vThreshold256);
    unsigned long long int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE256)/(int)sizeof(uint16_t);x++) {
      uint16_t u =  ((vector256_t)vThresholdMask).u16[x];
      if (u != (uint16_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====\n");
#endif
    return result;
  }
}

#endif




#ifdef __SSE2__

/**
 * SSE2 alignment init read function : modify the read length when needed (but must not be changed too frequently).
 * @param readlength gives the read length (number of nucleotides inside the read)
 */

void alignment_sse2__setlength_mono(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+15) * 2;

  /* allocating/reallocating mask table */
  if (vMsk128unaligned)
    free(vMsk128unaligned);
  vMsk128unaligned = malloc(prlength * sizeof(VTYPE128) + 15);
  if (!vMsk128unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk128 = (void *) ((uintptr_t)(vMsk128unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk128[0] = EPI64X_TYPE(_mm_set)(0xff00000000000000LL,
                                    0x0000000000000000LL);
#ifdef DEBUG_SIMD
  {
    vector128_t Msk;
    Msk.v = vMsk128[0];
    fprintf(stderr,"[0]\t Msk:%.16llx%.16llx\n",(Msk.u64[1]),(Msk.u64[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk128[l] = vMsk128[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 15*2) {
        vMsk128[l] = SI128_TYPE(_mm_srli)(vMsk128[l],(1));
      } else {
        /* mask at the beginning */
        if (l <= 15*2) {
          vMsk128[l] = SI128_TYPE(_mm_or)(vMsk128[l-1],SI128_TYPE(_mm_srli)(vMsk128[l],(1)));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector128_t Msk;
      Msk.v = vMsk128[l];
      fprintf(stderr,"[0]\t Msk:%.16llx%.16llx\n",(Msk.u64[1]),(Msk.u64[0]));
    }
#endif
  }/* for l */
}


void alignment_sse2__setlength_pair(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+7) * 2;

  /* allocating/reallocating mask table */
  if (vMsk128unaligned)
    free(vMsk128unaligned);
  vMsk128unaligned = malloc(prlength * sizeof(VTYPE128) + 15);
  if (!vMsk128unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk128 = (void *) ((uintptr_t)(vMsk128unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk128[0] = EPI64X_TYPE(_mm_set)(0xff00000000000000LL,
                                    0xff00000000000000LL);
#ifdef DEBUG_SIMD
  {
    vector128_t Msk;
    Msk.v = vMsk128[0];
    fprintf(stderr,"[0]\t Msk:%.16llx,%.16llx\n",(Msk.u64[1]),(Msk.u64[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk128[l] = vMsk128[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 7*2) {
        vMsk128[l] = EPI64_TYPE(_mm_srli)(vMsk128[l],(1)*8);
      } else {
        /* mask at the beginning */
        if (l <= 7*2) {
          vMsk128[l] = SI128_TYPE(_mm_or)(vMsk128[l-1],EPI64_TYPE(_mm_srli)(vMsk128[l],(1)*8));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector128_t Msk;
      Msk.v = vMsk128[l];
      fprintf(stderr,"[0]\t Msk:%.16llx,%.16llx\n",(Msk.u64[1]),(Msk.u64[0]));
    }
#endif
  }/* for l */
}


void alignment_sse2__setlength_quad(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+3) * 2;

  /* allocating/reallocating mask table */
  if (vMsk128unaligned)
    free(vMsk128unaligned);
  vMsk128unaligned = malloc(prlength * sizeof(VTYPE128) + 15);
  if (!vMsk128unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk128 = (void *) ((uintptr_t)(vMsk128unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk128[0] = EPI32_TYPE(_mm_set)(0xff000000,
                                   0xff000000,
                                   0xff000000,
                                   0xff000000);
#ifdef DEBUG_SIMD
  {
    vector128_t Msk;
    Msk.v = vMsk128[0];
    fprintf(stderr,"[0]\t Msk:%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk128[l] = vMsk128[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 3*2) {
        vMsk128[l] = EPI32_TYPE(_mm_srli)(vMsk128[l],(1)*8);
      } else {
        /* mask at the beginning */
        if (l <= 3*2) {
          vMsk128[l] = SI128_TYPE(_mm_or)(vMsk128[l-1],EPI32_TYPE(_mm_srli)(vMsk128[l],(1)*8));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector128_t Msk;
      Msk.v = vMsk128[l];
      fprintf(stderr,"[0]\t Msk:%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
    }
#endif
  }/* for l */
}


void alignment_sse2__setlength_octa(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+1) * 2;

  /* allocating/reallocating mask table */
  if (vMsk128unaligned)
    free(vMsk128unaligned);
  vMsk128unaligned = malloc(prlength * sizeof(VTYPE128) + 15);
  if (!vMsk128unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk128 = (void *) ((uintptr_t)(vMsk128unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk128[0] = EPI16_TYPE(_mm_set)(0xff00,0xff00,
                                   0xff00,0xff00,
                                   0xff00,0xff00,
                                   0xff00,0xff00);
#ifdef DEBUG_SIMD
  {
    vector128_t Msk;
    Msk.v = vMsk128[0];
    fprintf(stderr,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk128[l] = vMsk128[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 1*2) {
        vMsk128[l] = EPI16_TYPE(_mm_srli)(vMsk128[l],(1)*8);
      } else {
        /* mask at the beginning */
        if (l <= 1*2) {
          vMsk128[l] = SI128_TYPE(_mm_or)(vMsk128[l-1],EPI16_TYPE(_mm_srli)(vMsk128[l],(1)*8));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector128_t Msk;
      Msk.v = vMsk128[l];
      fprintf(stderr,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
    }
#endif
  }/* for l */
}


/**
 * SSE2 alignment init function : fix the scoring system and the length of the reads (must be called once before aligning)
 * @param match      inits the match score vector
 * @param mismatch   inits the mismatch penalty vector (positive value only)
 * @param gapopen    inits the gap penalty vector (positive value only)
 * @param gapextends inits the gap penalty vector (positive value only)
 * @param threshold  inits the scoring threshold (positive value only)
 * @param length     fixes the length of the reads that will be treated : this value can be changed
 * @see alignment_sse2__setlength_mono @see alignment_sse2__setlength_pair @see alignment_sse2__setlength_quad @see alignment_sse2__setlength_octa
 *        (but must not be changed too frequently).
 */

void alignment_sse2__init_mono(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_sse2__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with SSE2 instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes128          =  EPI32_TYPE(_mm_set1)(1); /* not used */
  vBufferMask128    =  EPI64X_TYPE(_mm_set)(0LL, 3LL);
  vThreshold128     =  EPI8_TYPE(_mm_set1)(u_threshold);
  vIndelOpenS128    =  EPI8_TYPE(_mm_set1)(gapopen);
  vIndelExtendsS128 =  EPI8_TYPE(_mm_set1)(gapextends);
  vMatchS128        =  EPI8_TYPE(_mm_set1)(match);
  vMismatchS128     =  EPI8_TYPE(_mm_set1)(mismatch);

  alignment_sse2__setlength_mono(readlength);
}


void alignment_sse2__init_pair(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_sse2__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with SSE2 instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes128          =  EPI32_TYPE(_mm_set1)(1);
  vBufferMask128    =  EPI64X_TYPE(_mm_set)(3LL, 3LL);
  vThreshold128     =  EPI8_TYPE(_mm_set1)(u_threshold);
  vIndelOpenS128    =  EPI8_TYPE(_mm_set1)(gapopen);
  vIndelExtendsS128 =  EPI8_TYPE(_mm_set1)(gapextends);
  vMatchS128        =  EPI8_TYPE(_mm_set1)(match);
  vMismatchS128     =  EPI8_TYPE(_mm_set1)(mismatch);

  alignment_sse2__setlength_pair(readlength);
}


void alignment_sse2__init_quad(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_sse2__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with SSE2 instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes128          =  EPI32_TYPE(_mm_set1)(1);
  vBufferMask128    =  EPI32_TYPE(_mm_set1)(3);
  vThreshold128     =  EPI8_TYPE(_mm_set1)(u_threshold);
  vIndelOpenS128    =  EPI8_TYPE(_mm_set1)(gapopen);
  vIndelExtendsS128 =  EPI8_TYPE(_mm_set1)(gapextends);
  vMatchS128        =  EPI8_TYPE(_mm_set1)(match);
  vMismatchS128     =  EPI8_TYPE(_mm_set1)(mismatch);

  alignment_sse2__setlength_quad(readlength);
}


void alignment_sse2__init_octa(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_sse2__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with SSE2 instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }


  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes128          =  EPI16_TYPE(_mm_set1)(1);
  vBufferMask128    =  EPI16_TYPE(_mm_set1)(3);
  vThreshold128     =  EPI8_TYPE(_mm_set1)(u_threshold);
  vIndelOpenS128    =  EPI8_TYPE(_mm_set1)(gapopen);
  vIndelExtendsS128 =  EPI8_TYPE(_mm_set1)(gapextends);
  vMatchS128        =  EPI8_TYPE(_mm_set1)(match);
  vMismatchS128     =  EPI8_TYPE(_mm_set1)(mismatch);

  alignment_sse2__setlength_octa(readlength);
}


/**
 * SSE2 alignment align function : does a banded smith-waterman of the given read against two parts of the genome;
 * allows at most 1/2, 3/4, 7/8 or 15/16 indels on each side.
 * @param genome is the compressed genome (first nucleotide is the lower bit of the first byte)
 * @param pos_genome gives the list of positions (in term of nucleotides) to be aligned with the read
 *        (you must substract 1/2, 3/4, 7/8 or 15/16 potential indels according to the hit position)
 * @param read is the compressed read (first nucleotide is the lower bit of the first byte)
 * @return 0 if none is aligned to reach the given threshold (what should happened most of the time),
 *         or a bitmask 1<<(x) | 1<<(y) if x or y align correctly.
 */

unsigned int alignment_sse2__align_mono(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read) {
  VTYPE128 vA;
  VTYPE128 vB;

  VTYPE128 vMMax;
  VTYPE128 vM_old;
  VTYPE128 vM_old_old;
  VTYPE128 vI_old;

  unsigned char *             byte_pos_genome = genome + (pos_genome[0] >> 2);
  unsigned int                 sub_pos_genome = (pos_genome[0] & 3);

  vector128_t        vector_genome_buffer;
  unsigned int       vector_genome_buffer_nbnuc = 0;

  vector128_t        vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI128_TYPE(_mm_setzero)();
  vM_old     = SI128_TYPE(_mm_setzero)();
  vM_old_old = SI128_TYPE(_mm_setzero)();
  vI_old     = SI128_TYPE(_mm_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI128_TYPE(_mm_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_MONO128(sub_pos_genome,byte_pos_genome,
                        vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 16; d++) {
      VTYPE128 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_MONO128(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = SI128_TYPE(_mm_slli)(vB,(1));
      vB        = SI128_TYPE(_mm_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE128 vLB;
        NEXTGENOSEQ_NOSUB_MONO128(byte_pos_genome,
                                  vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = SI128_TYPE(_mm_slli)(vB,(1));
        vB  = SI128_TYPE(_mm_or)(vB,vLB);
      } else {
        VTYPE128 vLA;
        NEXTREADSEQ_MONO128(read,
                            vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = SI128_TYPE(_mm_slli)(vLA,(15));
        vA  = SI128_TYPE(_mm_srli)(vA,(1));
        vA  = SI128_TYPE(_mm_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector128_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.16llx%.16llx\n",(A.u64[1]),(A.u64[0]));
        fprintf(stderr,"[1]\t   B:%.16llx%.16llx\n",(B.u64[1]),(B.u64[0]));
      }
#endif


      VTYPE128 vM;
      /* b) compute the matching score */
      {
        VTYPE128 vM_ab_MatchMask = EPI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE128 vM_add = SI128_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS128);
        VTYPE128 vM_sub = SI128_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS128);

#ifdef DEBUG_SIMD
        {
          vector128_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.16llx%.16llx\n",(S_a.u64[1]),(S_a.u64[0]));
          fprintf(stderr,"[1]\t S_s:%.16llx%.16llx\n",(S_s.u64[1]),(S_s.u64[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector128_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.16llx%.16llx\n",(M_old_old.u64[1]),(M_old_old.u64[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector128_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.16llx%.16llx\n",(    M.u64[1]),(    M.u64[0]));
        fprintf(stderr,"[1]\t  oM:%.16llx%.16llx\n",(M_old.u64[1]),(M_old.u64[0]));
        fprintf(stderr,"[1]\t  oI:%.16llx%.16llx\n",(I_old.u64[1]),(I_old.u64[0]));
      }
#endif

      VTYPE128 vI;
      {
        /* shift */
        VTYPE128 vM_old_shifted;
        VTYPE128 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = SI128_TYPE(_mm_slli)(vM_old,(1));
          vI_old_shifted  = SI128_TYPE(_mm_slli)(vI_old,(1));
        } else {
          vM_old_shifted  = SI128_TYPE(_mm_srli)(vM_old,(1));
          vI_old_shifted  = SI128_TYPE(_mm_srli)(vI_old,(1));
        }
        VTYPE128 vI_old_merge = EPU8_TYPE(_mm_max)(vI_old,vI_old_shifted);
        VTYPE128 vM_old_merge = EPU8_TYPE(_mm_max)(vM_old,vM_old_shifted);
        vI                    = EPU8_TYPE(_mm_subs)(vI_old_merge,vIndelExtendsS128);
        VTYPE128 vIstart      = EPU8_TYPE(_mm_subs)(vM_old_merge,vIndelOpenS128);
        vI                    = EPU8_TYPE(_mm_max)(vI,vIstart);
        vM                    = EPU8_TYPE(_mm_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector128_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.16llx%.16llx\n",(M.u64[1]),(M.u64[0]));
        fprintf(stderr,"[1]\t>  I:%.16llx%.16llx\n",(I.u64[1]),(I.u64[0]));
      }
#endif

      vM    = SI128_TYPE(_mm_and)(vM,vMsk128[l]);
      vMMax = EPU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector128_t Msk,M,Max;
        Msk.v = vMsk128[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.16llx%.16llx\n",(Msk.u64[1]),(Msk.u64[0]));
        fprintf(stderr,"[1]\t>  M:%.16llx%.16llx\n",(  M.u64[1]),(  M.u64[0]));
        fprintf(stderr,"[1]\t>Max:%.16llx%.16llx\n",(Max.u64[1]),(Max.u64[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     --------------------------------\n");
#endif
    } /* l */
  }
  {
    VTYPE128 vThresholdMask = EPU8_TYPE(_mm_subs)(vMMax,vThreshold128);
    unsigned long long int result = 0;
    uint64_t u0 =  ((vector128_t)vThresholdMask).u64[0];
    uint64_t u1 =  ((vector128_t)vThresholdMask).u64[1];
    if ((u0 != (uint64_t) 0) || (u1 != (uint64_t) 0)) {
      result |= 1;
    }
#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     ================================\n");
#endif
    return result;
  }
}


unsigned int alignment_sse2__align_pair(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read) {
  VTYPE128 vA;
  VTYPE128 vB;

  VTYPE128 vMMax;
  VTYPE128 vM_old;
  VTYPE128 vM_old_old;
  VTYPE128 vI_old;

  unsigned char *             byte_pos_genome[2];
  unsigned int                 sub_pos_genome[2];
  {
    int d;
    for (d = 0; d < 2; d++) {
      byte_pos_genome[d] = genome + (pos_genome[d] >> 2);
       sub_pos_genome[d] = (pos_genome[d] & 3);
    }
  }

  vector128_t        vector_genome_buffer;
  vector128_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI128_TYPE(_mm_setzero)();

  vector128_t        vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI128_TYPE(_mm_setzero)();
  vM_old     = SI128_TYPE(_mm_setzero)();
  vM_old_old = SI128_TYPE(_mm_setzero)();
  vI_old     = SI128_TYPE(_mm_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI128_TYPE(_mm_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_PAIR128(sub_pos_genome,byte_pos_genome,
                        vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 8; d++) {
      VTYPE128 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_PAIR128(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = EPI64_TYPE(_mm_slli)(vB,(1)*8);
      vB        = SI128_TYPE(_mm_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE128 vLB;
        NEXTGENOSEQ_NOSUB_PAIR128(byte_pos_genome,
                                  vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = EPI64_TYPE(_mm_slli)(vB,(1)*8);
        vB  = SI128_TYPE(_mm_or)(vB,vLB);
      } else {
        VTYPE128 vLA;
        NEXTREADSEQ_PAIR128(read,
                            vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = EPI64_TYPE(_mm_slli)(vLA,(7)*8);
        vA  = EPI64_TYPE(_mm_srli)(vA,(1)*8);
        vA  = SI128_TYPE(_mm_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector128_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.16llx,%.16llx\n",(A.u64[1]),(A.u64[0]));
        fprintf(stderr,"[1]\t   B:%.16llx,%.16llx\n",(B.u64[1]),(B.u64[0]));
      }
#endif


      VTYPE128 vM;
      /* b) compute the matching score */
      {
        VTYPE128 vM_ab_MatchMask = EPI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE128 vM_add = SI128_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS128);
        VTYPE128 vM_sub = SI128_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS128);

#ifdef DEBUG_SIMD
        {
          vector128_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.16llx,%.16llx\n",(S_a.u64[1]),(S_a.u64[0]));
          fprintf(stderr,"[1]\t S_s:%.16llx,%.16llx\n",(S_s.u64[1]),(S_s.u64[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector128_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.16llx,%.16llx\n",(M_old_old.u64[1]),(M_old_old.u64[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector128_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.16llx,%.16llx\n",(    M.u64[1]),(    M.u64[0]));
        fprintf(stderr,"[1]\t  oM:%.16llx,%.16llx\n",(M_old.u64[1]),(M_old.u64[0]));
        fprintf(stderr,"[1]\t  oI:%.16llx,%.16llx\n",(I_old.u64[1]),(I_old.u64[0]));
      }
#endif

      VTYPE128 vI;
      {
        /* shift */
        VTYPE128 vM_old_shifted;
        VTYPE128 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = EPI64_TYPE(_mm_slli)(vM_old,(1)*8);
          vI_old_shifted  = EPI64_TYPE(_mm_slli)(vI_old,(1)*8);
        } else {
          vM_old_shifted  = EPI64_TYPE(_mm_srli)(vM_old,(1)*8);
          vI_old_shifted  = EPI64_TYPE(_mm_srli)(vI_old,(1)*8);
        }
        VTYPE128 vI_old_merge = EPU8_TYPE(_mm_max)(vI_old,vI_old_shifted);
        VTYPE128 vM_old_merge = EPU8_TYPE(_mm_max)(vM_old,vM_old_shifted);
        vI                    = EPU8_TYPE(_mm_subs)(vI_old_merge,vIndelExtendsS128);
        VTYPE128 vIstart      = EPU8_TYPE(_mm_subs)(vM_old_merge,vIndelOpenS128);
        vI                    = EPU8_TYPE(_mm_max)(vI,vIstart);
        vM                    = EPU8_TYPE(_mm_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector128_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.16llx,%.16llx\n",(M.u64[1]),(M.u64[0]));
        fprintf(stderr,"[1]\t>  I:%.16llx,%.16llx\n",(I.u64[1]),(I.u64[0]));
      }
#endif

      vM    = SI128_TYPE(_mm_and)(vM,vMsk128[l]);
      vMMax = EPU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector128_t Msk,M,Max;
        Msk.v = vMsk128[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.16llx,%.16llx\n",(Msk.u64[1]),(Msk.u64[0]));
        fprintf(stderr,"[1]\t>  M:%.16llx,%.16llx\n",(  M.u64[1]),(  M.u64[0]));
        fprintf(stderr,"[1]\t>Max:%.16llx,%.16llx\n",(Max.u64[1]),(Max.u64[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     ----------------,----------------\n");
#endif
    } /* l */
  }
  {
    VTYPE128 vThresholdMask = EPU8_TYPE(_mm_subs)(vMMax,vThreshold128);
    unsigned long long int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE128)/(int)sizeof(uint64_t);x++) {
      uint64_t u =  ((vector128_t)vThresholdMask).u64[x];
      if (u != (uint64_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ================,================\n");
#endif
    return result;
  }
}


unsigned int alignment_sse2__align_quad(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read) {
  VTYPE128 vA;
  VTYPE128 vB;

  VTYPE128 vMMax;
  VTYPE128 vM_old;
  VTYPE128 vM_old_old;
  VTYPE128 vI_old;

  unsigned char *             byte_pos_genome[4];
  unsigned int                 sub_pos_genome[4];
  {
    int d;
    for (d = 0; d < 4; d++) {
      byte_pos_genome[d] = genome + (pos_genome[d] >> 2);
       sub_pos_genome[d] = (pos_genome[d] & 3);
    }
  }

  vector128_t        vector_genome_buffer;
  vector128_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI128_TYPE(_mm_setzero)();

  vector128_t        vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI128_TYPE(_mm_setzero)();
  vM_old     = SI128_TYPE(_mm_setzero)();
  vM_old_old = SI128_TYPE(_mm_setzero)();
  vI_old     = SI128_TYPE(_mm_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI128_TYPE(_mm_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_QUAD128(sub_pos_genome,byte_pos_genome,
                        vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 4; d++) {
      VTYPE128 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_QUAD128(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = EPI32_TYPE(_mm_slli)(vB,(1)*8);
      vB        = SI128_TYPE(_mm_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE128 vLB;
        NEXTGENOSEQ_NOSUB_QUAD128(byte_pos_genome,
                                  vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = EPI32_TYPE(_mm_slli)(vB,(1)*8);
        vB  = SI128_TYPE(_mm_or)(vB,vLB);
      } else {
        VTYPE128 vLA;
        NEXTREADSEQ_QUAD128(read,
                            vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = EPI32_TYPE(_mm_slli)(vLA,(3)*8);
        vA  = EPI32_TYPE(_mm_srli)(vA,(1)*8);
        vA  = SI128_TYPE(_mm_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector128_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.8x,%.8x,%.8x,%.8x\n",(A.u32[3]),(A.u32[2]),(A.u32[1]),(A.u32[0]));
        fprintf(stderr,"[1]\t   B:%.8x,%.8x,%.8x,%.8x\n",(B.u32[3]),(B.u32[2]),(B.u32[1]),(B.u32[0]));
      }
#endif


      VTYPE128 vM;
      /* b) compute the matching score */
      {
        VTYPE128 vM_ab_MatchMask = EPI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE128 vM_add = SI128_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS128);
        VTYPE128 vM_sub = SI128_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS128);

#ifdef DEBUG_SIMD
        {
          vector128_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.8x,%.8x,%.8x,%.8x\n",(S_a.u32[3]),(S_a.u32[2]),(S_a.u32[1]),(S_a.u32[0]));
          fprintf(stderr,"[1]\t S_s:%.8x,%.8x,%.8x,%.8x\n",(S_s.u32[3]),(S_s.u32[2]),(S_s.u32[1]),(S_s.u32[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector128_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.8x,%.8x,%.8x,%.8x\n",(M_old_old.u32[3]),(M_old_old.u32[2]),(M_old_old.u32[1]),(M_old_old.u32[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector128_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x\n",(    M.u32[3]),(    M.u32[2]),(    M.u32[1]),(    M.u32[0]));
        fprintf(stderr,"[1]\t  oM:%.8x,%.8x,%.8x,%.8x\n",(M_old.u32[3]),(M_old.u32[2]),(M_old.u32[1]),(M_old.u32[0]));
        fprintf(stderr,"[1]\t  oI:%.8x,%.8x,%.8x,%.8x\n",(I_old.u32[3]),(I_old.u32[2]),(I_old.u32[1]),(I_old.u32[0]));
      }
#endif

      VTYPE128 vI;
      {
        /* shift */
        VTYPE128 vM_old_shifted;
        VTYPE128 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = EPI32_TYPE(_mm_slli)(vM_old,(1)*8);
          vI_old_shifted  = EPI32_TYPE(_mm_slli)(vI_old,(1)*8);
        } else {
          vM_old_shifted  = EPI32_TYPE(_mm_srli)(vM_old,(1)*8);
          vI_old_shifted  = EPI32_TYPE(_mm_srli)(vI_old,(1)*8);
        }
        VTYPE128 vI_old_merge = EPU8_TYPE(_mm_max)(vI_old,vI_old_shifted);
        VTYPE128 vM_old_merge = EPU8_TYPE(_mm_max)(vM_old,vM_old_shifted);
        vI                    = EPU8_TYPE(_mm_subs)(vI_old_merge,vIndelExtendsS128);
        VTYPE128 vIstart      = EPU8_TYPE(_mm_subs)(vM_old_merge,vIndelOpenS128);
        vI                    = EPU8_TYPE(_mm_max)(vI,vIstart);
        vM                    = EPU8_TYPE(_mm_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector128_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x\n",(M.u32[3]),(M.u32[2]),(M.u32[1]),(M.u32[0]));
        fprintf(stderr,"[1]\t>  I:%.8x,%.8x,%.8x,%.8x\n",(I.u32[3]),(I.u32[2]),(I.u32[1]),(I.u32[0]));
      }
#endif

      vM    = SI128_TYPE(_mm_and)(vM,vMsk128[l]);
      vMMax = EPU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector128_t Msk,M,Max;
        Msk.v = vMsk128[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
        fprintf(stderr,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x\n",(  M.u32[3]),(  M.u32[2]),(  M.u32[1]),(  M.u32[0]));
        fprintf(stderr,"[1]\t>Max:%.8x,%.8x,%.8x,%.8x\n",(Max.u32[3]),(Max.u32[2]),(Max.u32[1]),(Max.u32[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     --------,--------,--------,--------\n");
#endif
    } /* l */
  }
  {
    VTYPE128 vThresholdMask = EPU8_TYPE(_mm_subs)(vMMax,vThreshold128);
    unsigned long long int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE128)/(int)sizeof(uint32_t);x++) {
      uint32_t u =  ((vector128_t)vThresholdMask).u32[x];
      if (u != (uint32_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ========,========,========,========\n");
#endif
    return result;
  }
}


unsigned int alignment_sse2__align_octa(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read) {
  VTYPE128 vA;
  VTYPE128 vB;

  VTYPE128 vMMax;
  VTYPE128 vM_old;
  VTYPE128 vM_old_old;
  VTYPE128 vI_old;

  unsigned char *             byte_pos_genome[8];
  unsigned int                 sub_pos_genome[8];
  {
    int d;
    for (d = 0; d < 8; d++) {
      byte_pos_genome[d] = genome + (pos_genome[d] >> 2);
       sub_pos_genome[d] = (pos_genome[d] & 3);
    }
  }

  vector128_t        vector_genome_buffer;
  vector128_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI128_TYPE(_mm_setzero)();

  vector128_t        vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI128_TYPE(_mm_setzero)();
  vM_old     = SI128_TYPE(_mm_setzero)();
  vM_old_old = SI128_TYPE(_mm_setzero)();
  vI_old     = SI128_TYPE(_mm_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI128_TYPE(_mm_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_OCTA128(sub_pos_genome,byte_pos_genome,
                        vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 2; d++) {
      VTYPE128 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_OCTA128(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = EPI16_TYPE(_mm_slli)(vB,(1)*8);
      vB        = SI128_TYPE(_mm_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE128 vLB;
        NEXTGENOSEQ_NOSUB_OCTA128(byte_pos_genome,
                                  vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = EPI16_TYPE(_mm_slli)(vB,(1)*8);
        vB  = SI128_TYPE(_mm_or)(vB,vLB);
      } else {
        VTYPE128 vLA;
        NEXTREADSEQ_OCTA128(read,
                            vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = EPI16_TYPE(_mm_slli)(vLA,(1)*8);
        vA  = EPI16_TYPE(_mm_srli)(vA,(1)*8);
        vA  = SI128_TYPE(_mm_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector128_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(A.u16[7]),(A.u16[6]),(A.u16[5]),(A.u16[4]),(A.u16[3]),(A.u16[2]),(A.u16[1]),(A.u16[0]));
        fprintf(stderr,"[1]\t   B:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(B.u16[7]),(B.u16[6]),(B.u16[5]),(B.u16[4]),(B.u16[3]),(B.u16[2]),(B.u16[1]),(B.u16[0]));
      }
#endif


      VTYPE128 vM;
      /* b) compute the matching score */
      {
        VTYPE128 vM_ab_MatchMask = EPI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE128 vM_add = SI128_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS128);
        VTYPE128 vM_sub = SI128_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS128);

#ifdef DEBUG_SIMD
        {
          vector128_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(S_a.u16[7]),(S_a.u16[6]),(S_a.u16[5]),(S_a.u16[4]),(S_a.u16[3]),(S_a.u16[2]),(S_a.u16[1]),(S_a.u16[0]));
          fprintf(stderr,"[1]\t S_s:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(S_s.u16[7]),(S_s.u16[6]),(S_s.u16[5]),(S_s.u16[4]),(S_s.u16[3]),(S_s.u16[2]),(S_s.u16[1]),(S_s.u16[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector128_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M_old_old.u16[7]),(M_old_old.u16[6]),(M_old_old.u16[5]),(M_old_old.u16[4]),(M_old_old.u16[3]),(M_old_old.u16[2]),(M_old_old.u16[1]),(M_old_old.u16[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector128_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(    M.u16[7]),(    M.u16[6]),(    M.u16[5]),(    M.u16[4]),(    M.u16[3]),(    M.u16[2]),(    M.u16[1]),(    M.u16[0]));
        fprintf(stderr,"[1]\t  oM:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M_old.u16[7]),(M_old.u16[6]),(M_old.u16[5]),(M_old.u16[4]),(M_old.u16[3]),(M_old.u16[2]),(M_old.u16[1]),(M_old.u16[0]));
        fprintf(stderr,"[1]\t  oI:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(I_old.u16[7]),(I_old.u16[6]),(I_old.u16[5]),(I_old.u16[4]),(I_old.u16[3]),(I_old.u16[2]),(I_old.u16[1]),(I_old.u16[0]));
      }
#endif

      VTYPE128 vI;
      {
        /* shift */
        VTYPE128 vM_old_shifted;
        VTYPE128 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = EPI16_TYPE(_mm_slli)(vM_old,(1)*8);
          vI_old_shifted  = EPI16_TYPE(_mm_slli)(vI_old,(1)*8);
        } else {
          vM_old_shifted  = EPI16_TYPE(_mm_srli)(vM_old,(1)*8);
          vI_old_shifted  = EPI16_TYPE(_mm_srli)(vI_old,(1)*8);
        }
        VTYPE128 vI_old_merge = EPU8_TYPE(_mm_max)(vI_old,vI_old_shifted);
        VTYPE128 vM_old_merge = EPU8_TYPE(_mm_max)(vM_old,vM_old_shifted);
        vI                    = EPU8_TYPE(_mm_subs)(vI_old_merge,vIndelExtendsS128);
        VTYPE128 vIstart      = EPU8_TYPE(_mm_subs)(vM_old_merge,vIndelOpenS128);
        vI                    = EPU8_TYPE(_mm_max)(vI,vIstart);
        vM                    = EPU8_TYPE(_mm_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector128_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M.u16[7]),(M.u16[6]),(M.u16[5]),(M.u16[4]),(M.u16[3]),(M.u16[2]),(M.u16[1]),(M.u16[0]));
        fprintf(stderr,"[1]\t>  I:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(I.u16[7]),(I.u16[6]),(I.u16[5]),(I.u16[4]),(I.u16[3]),(I.u16[2]),(I.u16[1]),(I.u16[0]));
      }
#endif

      vM    = SI128_TYPE(_mm_and)(vM,vMsk128[l]);
      vMMax = EPU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector128_t Msk,M,Max;
        Msk.v = vMsk128[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
        fprintf(stderr,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(  M.u16[7]),(  M.u16[6]),(  M.u16[5]),(  M.u16[4]),(  M.u16[3]),(  M.u16[2]),(  M.u16[1]),(  M.u16[0]));
        fprintf(stderr,"[1]\t>Max:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Max.u16[7]),(Max.u16[6]),(Max.u16[5]),(Max.u16[4]),(Max.u16[3]),(Max.u16[2]),(Max.u16[1]),(Max.u16[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     ----,----,----,----,----,----,----,----\n");
#endif
    } /* l */
  }
  {
    VTYPE128 vThresholdMask = EPU8_TYPE(_mm_subs)(vMMax,vThreshold128);
    unsigned long long int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE128)/(int)sizeof(uint16_t);x++) {
      uint16_t u =  ((vector128_t)vThresholdMask).u16[x];
      if (u != (uint16_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ====,====,====,====,====,====,====,====\n");
#endif
    return result;
  }
}

#endif




#ifdef __SSE__

/**
 * SSE alignment init read function : modify the read length when needed (but must not be changed too frequently).
 * @param readlength gives the read length (number of nucleotides inside the read)
 */

void alignment_sse__setlength_mono(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+7) * 2;

  /* allocating/reallocating mask table */
  if (vMsk64unaligned)
    free(vMsk64unaligned);
  vMsk64unaligned = malloc(prlength * sizeof(VTYPE64) + 15);
  if (!vMsk64unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk64 = (void *) ((uintptr_t)(vMsk64unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk64[0] = PI32_TYPE(_mm_set)(0xff000000,
                                 0x00000000);
#ifdef DEBUG_SIMD
  {
    vector64_t Msk;
    Msk.v = vMsk64[0];
    fprintf(stderr,"[0]\t Msk:%.16llx\n",(Msk.u64[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk64[l] = vMsk64[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 7*2) {
        vMsk64[l] = SI64_TYPE(_mm_srli)(vMsk64[l],(1)*8);
      } else {
        /* mask at the beginning */
        if (l <= 7*2) {
          vMsk64[l] = SI64_TYPE(_mm_or)(vMsk64[l-1],SI64_TYPE(_mm_srli)(vMsk64[l],(1)*8));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector64_t Msk;
      Msk.v = vMsk64[l];
      fprintf(stderr,"[0]\t Msk:%.16llx\n",(Msk.u64[0]));
    }
#endif
  }/* for l */
  _mm_empty();
}


void alignment_sse__setlength_pair(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+3) * 2;

  /* allocating/reallocating mask table */
  if (vMsk64unaligned)
    free(vMsk64unaligned);
  vMsk64unaligned = malloc(prlength * sizeof(VTYPE64) + 15);
  if (!vMsk64unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk64 = (void *) ((uintptr_t)(vMsk64unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk64[0] = PI32_TYPE(_mm_set)(0xff000000,
                                 0xff000000);
#ifdef DEBUG_SIMD
  {
    vector64_t Msk;
    Msk.v = vMsk64[0];
    fprintf(stderr,"[0]\t Msk:%.8x,%.8x\n",(Msk.u32[1]),(Msk.u32[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk64[l] = vMsk64[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 3*2) {
        vMsk64[l] = PI32_TYPE(_mm_srli)(vMsk64[l],(1)*8);
      } else {
        /* mask at the beginning */
        if (l <= 3*2) {
          vMsk64[l] = SI64_TYPE(_mm_or)(vMsk64[l-1],PI32_TYPE(_mm_srli)(vMsk64[l],(1)*8));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector64_t Msk;
      Msk.v = vMsk64[l];
      fprintf(stderr,"[0]\t Msk:%.8x,%.8x\n",(Msk.u32[1]),(Msk.u32[0]));
    }
#endif
  }/* for l */
  _mm_empty();
}


void alignment_sse__setlength_quad(const unsigned int readlength) {

  /* masking table to keep only good diagonals */
  prlength = (readlength+1) * 2;

  /* allocating/reallocating mask table */
  if (vMsk64unaligned)
    free(vMsk64unaligned);
  vMsk64unaligned = malloc(prlength * sizeof(VTYPE64) + 15);
  if (!vMsk64unaligned) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }
  vMsk64 = (void *) ((uintptr_t)(vMsk64unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk64[0] = PI16_TYPE(_mm_set)(0xff00,0xff00,
                                 0xff00,0xff00);
#ifdef DEBUG_SIMD
  {
    vector64_t Msk;
    Msk.v = vMsk64[0];
    fprintf(stderr,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
  }
#endif
  unsigned int l;
  for (l = 1; l < prlength; l++) {
    /* middle mask */
    vMsk64[l] = vMsk64[l - 1];
    if (!(l & 1)) {
      /* mask at the end */
      if (l >= prlength - 1*2) {
        vMsk64[l] = PI16_TYPE(_mm_srli)(vMsk64[l],(1)*8);
      } else {
        /* mask at the beginning */
        if (l <= 1*2) {
          vMsk64[l] = SI64_TYPE(_mm_or)(vMsk64[l-1],PI16_TYPE(_mm_srli)(vMsk64[l],(1)*8));
        }
      }
    }
#ifdef DEBUG_SIMD
    {
      vector64_t Msk;
      Msk.v = vMsk64[l];
      fprintf(stderr,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
    }
#endif
  }/* for l */
  _mm_empty();
}


/**
 * SSE alignment init function : fix the scoring system and the length of the reads (must be called once before aligning)
 * @param match      inits the match score vector
 * @param mismatch   inits the mismatch penalty vector (positive value only)
 * @param gapopen    inits the gap penalty vector (positive value only)
 * @param gapextends inits the gap penalty vector (positive value only)
 * @param threshold  inits the scoring threshold (positive value only)
 * @param length     fixes the length of the reads that will be treated : this value can be changed
 * @see alignment_sse__setlength_mono @see alignment_sse__setlength_pair @see alignment_sse__setlength_quad
 *        (but must not be changed too frequently).
 */

void alignment_sse__init_mono(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_sse__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with SSE instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes64          =  PI32_TYPE(_mm_set)(1, 1); /* not used */
  vBufferMask64    =  PI32_TYPE(_mm_set)(0, 3);
  vThreshold64     =  PI8_TYPE(_mm_set1)(u_threshold);
  vIndelOpenS64    =  PI8_TYPE(_mm_set1)(gapopen);
  vIndelExtendsS64 =  PI8_TYPE(_mm_set1)(gapextends);
  vMatchS64        =  PI8_TYPE(_mm_set1)(match);
  vMismatchS64     =  PI8_TYPE(_mm_set1)(mismatch);
  _mm_empty();
  alignment_sse__setlength_mono(readlength);
}


void alignment_sse__init_pair(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_sse__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with SSE instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes64          =  PI32_TYPE(_mm_set1)(1);
  vBufferMask64    =  PI32_TYPE(_mm_set1)(3);
  vThreshold64     =  PI8_TYPE(_mm_set1)(u_threshold);
  vIndelOpenS64    =  PI8_TYPE(_mm_set1)(gapopen);
  vIndelExtendsS64 =  PI8_TYPE(_mm_set1)(gapextends);
  vMatchS64        =  PI8_TYPE(_mm_set1)(match);
  vMismatchS64     =  PI8_TYPE(_mm_set1)(mismatch);
  _mm_empty();
  alignment_sse__setlength_pair(readlength);
}


void alignment_sse__init_quad(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_sse__compatible_proc()) {
    fprintf(stderr,"\033[31;1m");
    fprintf(stderr,"\nCPU is not compatible with SSE instructions set.\nExiting.\n");
    fprintf(stderr,"\033[0m\n");
    exit(1);
  }

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes64          =  PI16_TYPE(_mm_set1)(1);
  vBufferMask64    =  PI16_TYPE(_mm_set1)(3);
  vThreshold64     =  PI8_TYPE(_mm_set1)(u_threshold);
  vIndelOpenS64    =  PI8_TYPE(_mm_set1)(gapopen);
  vIndelExtendsS64 =  PI8_TYPE(_mm_set1)(gapextends);
  vMatchS64        =  PI8_TYPE(_mm_set1)(match);
  vMismatchS64     =  PI8_TYPE(_mm_set1)(mismatch);
  _mm_empty();
  alignment_sse__setlength_quad(readlength);
}


/**
 * SSE alignment align function : does a banded smith-waterman of the given read against two parts of the genome;
 * allows at most 1/2, 3/4 or 7/8 indels on each side.
 * @param genome is the compressed genome (first nucleotide is the lower bit of the first byte)
 * @param pos_genome gives the list of positions (in term of nucleotides) to be aligned with the read
 *        (you must substract 1/2, 3/4 or 7/8 potential indels according to the hit position)
 * @param read is the compressed read (first nucleotide is the lower bit of the first byte)
 * @return 0 if none is aligned to reach the given threshold (what should happened most of the time),
 *         or a bitmask 1<<(x) | 1<<(y) if x or y align correctly.
 */

unsigned int alignment_sse__align_mono(unsigned char * genome,
                                       int * pos_genome,
                                       unsigned char * read) {
  VTYPE64 vA;
  VTYPE64 vB;

  VTYPE64 vMMax;
  VTYPE64 vM_old;
  VTYPE64 vM_old_old;
  VTYPE64 vI_old;

  unsigned char *             byte_pos_genome = genome + (pos_genome[0] >> 2);
  unsigned int                 sub_pos_genome = (pos_genome[0] & 3);

  vector64_t         vector_genome_buffer;
  unsigned int       vector_genome_buffer_nbnuc = 0;

  vector64_t         vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI64_TYPE(_mm_setzero)();
  vM_old     = SI64_TYPE(_mm_setzero)();
  vM_old_old = SI64_TYPE(_mm_setzero)();
  vI_old     = SI64_TYPE(_mm_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI64_TYPE(_mm_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_MONO64(sub_pos_genome,byte_pos_genome,
                       vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 8; d++) {
      VTYPE64 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_MONO64(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = SI64_TYPE(_mm_slli)(vB,(1)*8);
      vB        = SI64_TYPE(_mm_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE64 vLB;
        NEXTGENOSEQ_NOSUB_MONO64(byte_pos_genome,
                                 vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = SI64_TYPE(_mm_slli)(vB,(1)*8);
        vB  = SI64_TYPE(_mm_or)(vB,vLB);
      } else {
        VTYPE64 vLA;
        NEXTREADSEQ_MONO64(read,
                           vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = SI64_TYPE(_mm_slli)(vLA,(7)*8);
        vA  = SI64_TYPE(_mm_srli)(vA,(1)*8);
        vA  = SI64_TYPE(_mm_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector64_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.16llx\n",(A.u64[0]));
        fprintf(stderr,"[1]\t   B:%.16llx\n",(B.u64[0]));
      }
#endif


      VTYPE64 vM;
      /* b) compute the matching score */
      {
        VTYPE64 vM_ab_MatchMask = PI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE64 vM_add = SI64_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS64);
        VTYPE64 vM_sub = SI64_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS64);

#ifdef DEBUG_SIMD
        {
          vector64_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.16llx\n",(S_a.u64[0]));
          fprintf(stderr,"[1]\t S_s:%.16llx\n",(S_s.u64[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector64_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.16llx\n",(M_old_old.u64[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = PU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = PU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector64_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.16llx\n",(    M.u64[0]));
        fprintf(stderr,"[1]\t  oM:%.16llx\n",(M_old.u64[0]));
        fprintf(stderr,"[1]\t  oI:%.16llx\n",(I_old.u64[0]));
      }
#endif

      VTYPE64 vI;
      {
        /* shift */
        VTYPE64 vM_old_shifted;
        VTYPE64 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = SI64_TYPE(_mm_slli)(vM_old,(1)*8);
          vI_old_shifted  = SI64_TYPE(_mm_slli)(vI_old,(1)*8);
        } else {
          vM_old_shifted  = SI64_TYPE(_mm_srli)(vM_old,(1)*8);
          vI_old_shifted  = SI64_TYPE(_mm_srli)(vI_old,(1)*8);
        }
        VTYPE64 vI_old_merge  = PU8_TYPE(_mm_max)(vI_old,vI_old_shifted);
        VTYPE64 vM_old_merge  = PU8_TYPE(_mm_max)(vM_old,vM_old_shifted);
        vI                    = PU8_TYPE(_mm_subs)(vI_old_merge,vIndelExtendsS64);
        VTYPE64 vIstart       = PU8_TYPE(_mm_subs)(vM_old_merge,vIndelOpenS64);
        vI                    = PU8_TYPE(_mm_max)(vI,vIstart);
        vM                    = PU8_TYPE(_mm_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector64_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.16llx\n",(M.u64[0]));
        fprintf(stderr,"[1]\t>  I:%.16llx\n",(I.u64[0]));
      }
#endif

      vM    = SI64_TYPE(_mm_and)(vM,vMsk64[l]);
      vMMax = PU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector64_t Msk,M,Max;
        Msk.v = vMsk64[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.16llx\n",(Msk.u64[0]));
        fprintf(stderr,"[1]\t>  M:%.16llx\n",(  M.u64[0]));
        fprintf(stderr,"[1]\t>Max:%.16llx\n",(Max.u64[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     ----------------\n");
#endif
    } /* l */
  }
  {
    VTYPE64 vThresholdMask = PU8_TYPE(_mm_subs)(vMMax,vThreshold64);
    unsigned long long int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE64)/(int)sizeof(uint64_t);x++) {
      uint64_t u =  ((vector64_t)vThresholdMask).u64[x];
      if (u != (uint64_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ================\n");
#endif
    _mm_empty();
    return result;
  }
}


unsigned int alignment_sse__align_pair(unsigned char * genome,
                                       int * pos_genome,
                                       unsigned char * read) {
  VTYPE64 vA;
  VTYPE64 vB;

  VTYPE64 vMMax;
  VTYPE64 vM_old;
  VTYPE64 vM_old_old;
  VTYPE64 vI_old;

  unsigned char *             byte_pos_genome[2];
  unsigned int                 sub_pos_genome[2];
  {
    int d;
    for (d = 0; d < 2; d++) {
      byte_pos_genome[d] = genome + (pos_genome[d] >> 2);
       sub_pos_genome[d] = (pos_genome[d] & 3);
    }
  }

  vector64_t         vector_genome_buffer;
  vector64_t         vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI64_TYPE(_mm_setzero)();

  vector64_t         vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI64_TYPE(_mm_setzero)();
  vM_old     = SI64_TYPE(_mm_setzero)();
  vM_old_old = SI64_TYPE(_mm_setzero)();
  vI_old     = SI64_TYPE(_mm_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI64_TYPE(_mm_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_PAIR64(sub_pos_genome,byte_pos_genome,
                       vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 4; d++) {
      VTYPE64 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_PAIR64(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = PI32_TYPE(_mm_slli)(vB,(1)*8);
      vB        = SI64_TYPE(_mm_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE64 vLB;
        NEXTGENOSEQ_NOSUB_PAIR64(byte_pos_genome,
                                 vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = PI32_TYPE(_mm_slli)(vB,(1)*8);
        vB  = SI64_TYPE(_mm_or)(vB,vLB);
      } else {
        VTYPE64 vLA;
        NEXTREADSEQ_PAIR64(read,
                           vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = PI32_TYPE(_mm_slli)(vLA,(3)*8);
        vA  = PI32_TYPE(_mm_srli)(vA,(1)*8);
        vA  = SI64_TYPE(_mm_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector64_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.8x,%.8x\n",(A.u32[1]),(A.u32[0]));
        fprintf(stderr,"[1]\t   B:%.8x,%.8x\n",(B.u32[1]),(B.u32[0]));
      }
#endif


      VTYPE64 vM;
      /* b) compute the matching score */
      {
        VTYPE64 vM_ab_MatchMask = PI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE64 vM_add = SI64_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS64);
        VTYPE64 vM_sub = SI64_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS64);

#ifdef DEBUG_SIMD
        {
          vector64_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.8x,%.8x\n",(S_a.u32[1]),(S_a.u32[0]));
          fprintf(stderr,"[1]\t S_s:%.8x,%.8x\n",(S_s.u32[1]),(S_s.u32[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector64_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.8x,%.8x\n",(M_old_old.u32[1]),(M_old_old.u32[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = PU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = PU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector64_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.8x,%.8x\n",(    M.u32[1]),(    M.u32[0]));
        fprintf(stderr,"[1]\t  oM:%.8x,%.8x\n",(M_old.u32[1]),(M_old.u32[0]));
        fprintf(stderr,"[1]\t  oI:%.8x,%.8x\n",(I_old.u32[1]),(I_old.u32[0]));
      }
#endif

      VTYPE64 vI;
      {
        /* shift */
        VTYPE64 vM_old_shifted;
        VTYPE64 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = PI32_TYPE(_mm_slli)(vM_old,(1)*8);
          vI_old_shifted  = PI32_TYPE(_mm_slli)(vI_old,(1)*8);
        } else {
          vM_old_shifted  = PI32_TYPE(_mm_srli)(vM_old,(1)*8);
          vI_old_shifted  = PI32_TYPE(_mm_srli)(vI_old,(1)*8);
        }
        VTYPE64 vI_old_merge  = PU8_TYPE(_mm_max)(vI_old,vI_old_shifted);
        VTYPE64 vM_old_merge  = PU8_TYPE(_mm_max)(vM_old,vM_old_shifted);
        vI                    = PU8_TYPE(_mm_subs)(vI_old_merge,vIndelExtendsS64);
        VTYPE64 vIstart       = PU8_TYPE(_mm_subs)(vM_old_merge,vIndelOpenS64);
        vI                    = PU8_TYPE(_mm_max)(vI,vIstart);
        vM                    = PU8_TYPE(_mm_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector64_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.8x,%.8x\n",(M.u32[1]),(M.u32[0]));
        fprintf(stderr,"[1]\t>  I:%.8x,%.8x\n",(I.u32[1]),(I.u32[0]));
      }
#endif

      vM    = SI64_TYPE(_mm_and)(vM,vMsk64[l]);
      vMMax = PU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector64_t Msk,M,Max;
        Msk.v = vMsk64[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.8x,%.8x\n",(Msk.u32[1]),(Msk.u32[0]));
        fprintf(stderr,"[1]\t>  M:%.8x,%.8x\n",(  M.u32[1]),(  M.u32[0]));
        fprintf(stderr,"[1]\t>Max:%.8x,%.8x\n",(Max.u32[1]),(Max.u32[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     --------,--------\n");
#endif
    } /* l */
  }
  {
    VTYPE64 vThresholdMask = PU8_TYPE(_mm_subs)(vMMax,vThreshold64);
    unsigned long long int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE64)/(int)sizeof(uint32_t);x++) {
      uint32_t u =  ((vector64_t)vThresholdMask).u32[x];
      if (u != (uint32_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ========,========\n");
#endif
    _mm_empty();
    return result;
  }
}


unsigned int alignment_sse__align_quad(unsigned char * genome,
                                       int * pos_genome,
                                       unsigned char * read) {
  VTYPE64 vA;
  VTYPE64 vB;

  VTYPE64 vMMax;
  VTYPE64 vM_old;
  VTYPE64 vM_old_old;
  VTYPE64 vI_old;

  unsigned char *             byte_pos_genome[4];
  unsigned int                 sub_pos_genome[4];
  {
    int d;
    for (d = 0; d < 4; d++) {
      byte_pos_genome[d] = genome + (pos_genome[d] >> 2);
       sub_pos_genome[d] = (pos_genome[d] & 3);
    }
  }

  vector64_t         vector_genome_buffer;
  vector64_t         vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI64_TYPE(_mm_setzero)();

  vector64_t         vector_read_buffer;
  unsigned int       vector_read_buffer_nbnuc = 0;


  vMMax      = SI64_TYPE(_mm_setzero)();
  vM_old     = SI64_TYPE(_mm_setzero)();
  vM_old_old = SI64_TYPE(_mm_setzero)();
  vI_old     = SI64_TYPE(_mm_setzero)();

  /* init vA : read sequence diagonal */
  {
    vA       = SI64_TYPE(_mm_setzero)();
  }

  /* init vB : genome sequence diagonal */
  {
    int d;
    NEXTGENOSEQ_QUAD64(sub_pos_genome,byte_pos_genome,
                       vector_genome_buffer_nbnuc,vector_genome_buffer,vB);
    for (d = 1; d < 2; d++) {
      VTYPE64 vLB;
      NEXTGENOSEQ_NOSUB_NOUP_QUAD64(vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
      vB        = PI16_TYPE(_mm_slli)(vB,(1)*8);
      vB        = SI64_TYPE(_mm_or)(vB,vLB);
    }
  }


  /* main loop */
  {
    unsigned int l;
    for (l = 0; l < prlength; l++) {

      /* a) get the nucleotides to be compared on the diagonal */
      if (l & 1) {
        VTYPE64 vLB;
        NEXTGENOSEQ_NOSUB_QUAD64(byte_pos_genome,
                                 vector_genome_buffer_nbnuc,vector_genome_buffer,vLB);
        vB  = PI16_TYPE(_mm_slli)(vB,(1)*8);
        vB  = SI64_TYPE(_mm_or)(vB,vLB);
      } else {
        VTYPE64 vLA;
        NEXTREADSEQ_QUAD64(read,
                           vector_read_buffer_nbnuc,vector_read_buffer,vLA);
        vLA = PI16_TYPE(_mm_slli)(vLA,(1)*8);
        vA  = PI16_TYPE(_mm_srli)(vA,(1)*8);
        vA  = SI64_TYPE(_mm_or)(vA,vLA);
      }

#ifdef DEBUG_SIMD
      {
        vector64_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stderr,"[1]\t   A:%.4x,%.4x,%.4x,%.4x\n",(A.u16[3]),(A.u16[2]),(A.u16[1]),(A.u16[0]));
        fprintf(stderr,"[1]\t   B:%.4x,%.4x,%.4x,%.4x\n",(B.u16[3]),(B.u16[2]),(A.u16[1]),(A.u16[0]));
      }
#endif


      VTYPE64 vM;
      /* b) compute the matching score */
      {
        VTYPE64 vM_ab_MatchMask = PI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE64 vM_add = SI64_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS64);
        VTYPE64 vM_sub = SI64_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS64);

#ifdef DEBUG_SIMD
        {
          vector64_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stderr,"[1]\t S_a:%.4x,%.4x,%.4x,%.4x\n",(S_a.u16[3]),(S_a.u16[2]),(S_a.u16[1]),(S_a.u16[0]));
          fprintf(stderr,"[1]\t S_s:%.4x,%.4x,%.4x,%.4x\n",(S_s.u16[3]),(S_s.u16[2]),(S_s.u16[1]),(S_s.u16[0]));
        }
#endif


#ifdef DEBUG_SIMD
        {
          vector64_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stderr,"[1]\t ooM:%.4x,%.4x,%.4x,%.4x\n",(M_old_old.u16[3]),(M_old_old.u16[2]),(M_old_old.u16[1]),(M_old_old.u16[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = PU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = PU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG_SIMD
      {
        vector64_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stderr,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x\n",(    M.u16[3]),(    M.u16[2]),(    M.u16[1]),(    M.u16[0]));
        fprintf(stderr,"[1]\t  oM:%.4x,%.4x,%.4x,%.4x\n",(M_old.u16[3]),(M_old.u16[2]),(M_old.u16[1]),(M_old.u16[0]));
        fprintf(stderr,"[1]\t  oI:%.4x,%.4x,%.4x,%.4x\n",(I_old.u16[3]),(I_old.u16[2]),(I_old.u16[1]),(I_old.u16[0]));
      }
#endif

      VTYPE64 vI;
      {
        /* shift */
        VTYPE64 vM_old_shifted;
        VTYPE64 vI_old_shifted;
        if (l & 1) {
          vM_old_shifted  = PI16_TYPE(_mm_slli)(vM_old,(1)*8);
          vI_old_shifted  = PI16_TYPE(_mm_slli)(vI_old,(1)*8);
        } else {
          vM_old_shifted  = PI16_TYPE(_mm_srli)(vM_old,(1)*8);
          vI_old_shifted  = PI16_TYPE(_mm_srli)(vI_old,(1)*8);
        }
        VTYPE64 vI_old_merge  = PU8_TYPE(_mm_max)(vI_old,vI_old_shifted);
        VTYPE64 vM_old_merge  = PU8_TYPE(_mm_max)(vM_old,vM_old_shifted);
        vI                    = PU8_TYPE(_mm_subs)(vI_old_merge,vIndelExtendsS64);
        VTYPE64 vIstart       = PU8_TYPE(_mm_subs)(vM_old_merge,vIndelOpenS64);
        vI                    = PU8_TYPE(_mm_max)(vI,vIstart);
        vM                    = PU8_TYPE(_mm_max)(vM,vI);
      }

#ifdef DEBUG_SIMD
      {
        vector64_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stderr,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x\n",(M.u16[3]),(M.u16[2]),(M.u16[1]),(M.u16[0]));
        fprintf(stderr,"[1]\t>  I:%.4x,%.4x,%.4x,%.4x\n",(I.u16[3]),(I.u16[2]),(I.u16[1]),(I.u16[0]));
      }
#endif

      vM    = SI64_TYPE(_mm_and)(vM,vMsk64[l]);
      vMMax = PU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG_SIMD
      {
        vector64_t Msk,M,Max;
        Msk.v = vMsk64[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stderr,"[1]\t>Msk:%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
        fprintf(stderr,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x\n",(  M.u16[3]),(  M.u16[2]),(  M.u16[1]),(  M.u16[0]));
        fprintf(stderr,"[1]\t>Max:%.4x,%.4x,%.4x,%.4x\n",(Max.u16[3]),(Max.u16[2]),(Max.u16[1]),(Max.u16[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG_SIMD
      fprintf(stderr,"[1]\t     ----,----,----,----\n");
#endif
    } /* l */
  }
  {
    VTYPE64 vThresholdMask = PU8_TYPE(_mm_subs)(vMMax,vThreshold64);
    unsigned long long int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE64)/(int)sizeof(uint16_t);x++) {
      uint16_t u =  ((vector64_t)vThresholdMask).u16[x];
      if (u != (uint16_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG_SIMD
    fprintf(stderr,"[1]\t     ====,====,====,====\n");
#endif
    _mm_empty();
    return result;
  }
}

#endif
