#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "alignment_simd.h"

/*
 * #define DEBUG
 */

#ifdef __AVX2__

#include <immintrin.h>

#define EPU8_TYPE(ins)   ins##_epu8
#define EPI8_TYPE(ins)   ins##_epi8
#define EPI32_TYPE(ins)  ins##_epi32
#define EPI64_TYPE(ins)  ins##_epi64
#define SI256_TYPE(ins)  ins##_si256
#define VTYPE256            __m256i

/* data conversion union */
typedef union __attribute__((packed, aligned (16))) {
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
  /* __uint128_t u128[sizeof(VTYPE256)/sizeof(__uint128_t)];*/
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
  printf("* compatible avx2 ? %s\n", _bx & 1<<5 ? "yes":"no");
  return (_bx & 1<<5) != 0;
#else
  printf("* compatible avx2 ? no (32bits compiled)\n");
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
  printf("* compatible sse2 ? %s\n", _dx & 1<<26 ? "yes":"no");
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
  printf("* compatible sse ? %s\n", _dx & 1<<25 ? "yes":"no");
  return (_dx & 1<<25) != 0;
}


/*
 * macros and global variables (no need to export because not usefull ouside this .c file)
 */

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
    __u__ = EPI8_TYPE(_mm256_movemask)(EPI32_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                       \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 8; __d__++) {                                                                                                \
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
    out_vtype_vLB        = SI256_TYPE(_mm256_and)(inout_vector.v,vBufferMask256);                                                          \
    inout_vector.v       = EPI32_TYPE(_mm256_srli)(inout_vector.v,2);                                                                      \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm256_sub)(inout_nbnuc_vector.v,vOnes256);                                                          \
}

#define NEXTGENOSEQ_QUAD256(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                               \
    unsigned int __u__;                                                                                                                    \
    __u__ = EPI8_TYPE(_mm256_movemask)(EPI64_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                       \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 4; __d__++) {                                                                                                \
        if (__u__ & 1 << (8*__d__)) {                                                                                                      \
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
    __u__ = EPI8_TYPE(_mm256_movemask)(EPI64_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                       \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 2; __d__++) {                                                                                                \
        if (__u__ & 1 << (16*__d__)) {                                                                                                     \
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
    __u__ = EPI8_TYPE(_mm256_movemask)(EPI32_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                       \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 8; __d__++) {                                                                                                \
        if (__u__ & 1 << (4*__d__)) {                                                                                                      \
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
    __u__ = EPI8_TYPE(_mm256_movemask)(EPI64_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                       \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 4; __d__++) {                                                                                                \
        if (__u__ & 1 << (8*__d__)) {                                                                                                      \
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
    __u__ = EPI8_TYPE(_mm256_movemask)(EPI64_TYPE(_mm256_cmpeq)(inout_nbnuc_vector.v,SI256_TYPE(_mm256_setzero)()));                       \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 2; __d__++) {                                                                                                \
        if (__u__ & 1 << (16*__d__)) {                                                                                                     \
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

VTYPE256  vThreshold256     __attribute__ ((aligned (16))),
          vMatchS256        __attribute__ ((aligned (16))),
          vMismatchS256     __attribute__ ((aligned (16))),
          vIndelOpenS256    __attribute__ ((aligned (16))),
          vIndelExtendsS256 __attribute__ ((aligned (16))),
          vOnes256          __attribute__ ((aligned (16))),
          vBufferMask256    __attribute__ ((aligned (16))),
         *vMsk256           __attribute__ ((aligned (16)));

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
    __u__ = EPI8_TYPE(_mm_movemask)(EPI32_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI128_TYPE(_mm_setzero)()));                                \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 4; __d__++) {                                                                                                \
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
    out_vtype_vLB        = SI128_TYPE(_mm_and)(inout_vector.v,vBufferMask128);                                                             \
    inout_vector.v       = EPI32_TYPE(_mm_srli)(inout_vector.v,2);                                                                         \
    inout_nbnuc_vector.v = EPI32_TYPE(_mm_sub)(inout_nbnuc_vector.v,vOnes128);                                                             \
}

#define NEXTGENOSEQ_PAIR128(inout_subpos,inout_byte,                                                                                       \
                            inout_nbnuc_vector,inout_vector,out_vtype_vLB) {                                                               \
    unsigned int __u__;                                                                                                                    \
    __u__ = EPI8_TYPE(_mm_movemask)(EPI32_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI128_TYPE(_mm_setzero)()));                                \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 2; __d__++) {                                                                                                \
        if (__u__ & 1 << (8*__d__)) {                                                                                                      \
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
    __u__ = EPI8_TYPE(_mm_movemask)(EPI32_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI128_TYPE(_mm_setzero)()));                                \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 4; __d__++) {                                                                                                \
        if (__u__ & 1 << (4*__d__)) {                                                                                                      \
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
    __u__ = EPI8_TYPE(_mm_movemask)(EPI32_TYPE(_mm_cmpeq)(inout_nbnuc_vector.v,SI128_TYPE(_mm_setzero)()));                                \
    if (__u__) {                                                                                                                           \
      unsigned int __d__;                                                                                                                  \
      for (__d__ = 0; __d__ < 2; __d__++) {                                                                                                \
        if (__u__ & 1 << (8*__d__)) {                                                                                                      \
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
    out_vtype_vLA  = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                                     \
    inout_vector.v = PI16_TYPE(_mm_srli)(inout_vector.v,2);                                                                                \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTREADSEQ_PAIR64(inout_byte,inout_nbnuc,                                                                                         \
                           inout_vector,out_vtype_vLA) {                                                                                   \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = PI32_TYPE(_mm_set1)(*(uint32_t *)(inout_byte));                                                                     \
      inout_nbnuc    = 16;                                                                                                                 \
      inout_byte    += 4;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                                     \
    inout_vector.v = PI32_TYPE(_mm_srli)(inout_vector.v,2);                                                                                \
    inout_nbnuc--;                                                                                                                         \
}

#define NEXTREADSEQ_MONO64(inout_byte,inout_nbnuc,                                                                                         \
                           inout_vector,out_vtype_vLA) {                                                                                   \
    if (!inout_nbnuc) {                                                                                                                    \
      inout_vector.v = _mm_cvtsi64_m64(*((uint64_t *)(inout_byte)));                                                                       \
      inout_nbnuc    = 32;                                                                                                                 \
      inout_byte    += 8;                                                                                                                  \
    }                                                                                                                                      \
    out_vtype_vLA  = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                                     \
    inout_vector.v = SI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                                \
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
    out_vtype_vLB  = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                                     \
    inout_vector.v = SI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                                \
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
    out_vtype_vLB  = SI64_TYPE(_mm_and)(inout_vector.v,vBufferMask64);                                                                     \
    inout_vector.v = SI64_TYPE(_mm_srli)(inout_vector.v,2);                                                                                \
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
  vMsk256unaligned = malloc(prlength * sizeof(VTYPE256) + 15);
  if (!vMsk256unaligned) {
    printf("\033[31;1m");
    printf("\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    printf("\033[0m\n");
    exit(1);
  }
  vMsk256 = (void *) ((uintptr_t)(vMsk256unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk256[0] = EPI8_TYPE(_mm256_set)(0xff,0x00,0x00,0x00,
                                     0x00,0x00,0x00,0x00,
                                     0x00,0x00,0x00,0x00,
                                     0x00,0x00,0x00,0x00,
                                     0xff,0x00,0x00,0x00,
                                     0x00,0x00,0x00,0x00,
                                     0x00,0x00,0x00,0x00,
                                     0x00,0x00,0x00,0x00);
#ifdef DEBUG
  {
    vector256_t Msk;
    Msk.v = vMsk256[0];
    fprintf(stdout,"[0]\t Msk:%.16llx%.16llx,%.16llx%.16llx\n",(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
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
#ifdef DEBUG
    {
      vector256_t Msk;
      Msk.v = vMsk256[l];
      fprintf(stdout,"[0]\t Msk:%.16llx%.16llx,%.16llx%.16llx\n",(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
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
  vMsk256unaligned = malloc(prlength * sizeof(VTYPE256) + 15);
  if (!vMsk256unaligned) {
    printf("\033[31;1m");
    printf("\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    printf("\033[0m\n");
    exit(1);
  }
  vMsk256 = (void *) ((uintptr_t)(vMsk256unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk256[0] = EPI8_TYPE(_mm256_set)(0xff,0x00,0x00,0x00,
                                     0x00,0x00,0x00,0x00,
                                     0xff,0x00,0x00,0x00,
                                     0x00,0x00,0x00,0x00,
                                     0xff,0x00,0x00,0x00,
                                     0x00,0x00,0x00,0x00,
                                     0xff,0x00,0x00,0x00,
                                     0x00,0x00,0x00,0x00);
#ifdef DEBUG
  {
    vector256_t Msk;
    Msk.v = vMsk256[0];
    fprintf(stdout,"[0]\t Msk:%.16llx,%.16llx,%.16llx,%.16llx\n",(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
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
#ifdef DEBUG
    {
      vector256_t Msk;
      Msk.v = vMsk256[l];
      fprintf(stdout,"[0]\t Msk:%.16llx,%.16llx,%.16llx,%.16llx\n",(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
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
  vMsk256unaligned = malloc(prlength * sizeof(VTYPE256) + 15);
  if (!vMsk256unaligned) {
    printf("\033[31;1m");
    printf("\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    printf("\033[0m\n");
    exit(1);
  }
  vMsk256 = (void *) ((uintptr_t)(vMsk256unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk256[0] = EPI8_TYPE(_mm256_set)(0xff,0x00,0x00,0x00,
                                     0xff,0x00,0x00,0x00,
                                     0xff,0x00,0x00,0x00,
                                     0xff,0x00,0x00,0x00,
                                     0xff,0x00,0x00,0x00,
                                     0xff,0x00,0x00,0x00,
                                     0xff,0x00,0x00,0x00,
                                     0xff,0x00,0x00,0x00);
#ifdef DEBUG
  {
    vector256_t Msk;
    Msk.v = vMsk256[0];
    fprintf(stdout,"[0]\t Msk:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[7]),(Msk.u32[6]),(Msk.u32[5]),(Msk.u32[4]),(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
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
#ifdef DEBUG
    {
      vector256_t Msk;
      Msk.v = vMsk256[l];
      fprintf(stdout,"[0]\t Msk:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[7]),(Msk.u32[6]),(Msk.u32[5]),(Msk.u32[4]),(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
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
  vMsk256unaligned = malloc(prlength * sizeof(VTYPE256) + 15);
  if (!vMsk256unaligned) {
    printf("\033[31;1m");
    printf("\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    printf("\033[0m\n");
    exit(1);
  }
  vMsk256 = (void *) ((uintptr_t)(vMsk256unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk256[0] = EPI8_TYPE(_mm256_set)(0xff,0x00,0xff,0x00,
                                     0xff,0x00,0xff,0x00,
                                     0xff,0x00,0xff,0x00,
                                     0xff,0x00,0xff,0x00,
                                     0xff,0x00,0xff,0x00,
                                     0xff,0x00,0xff,0x00,
                                     0xff,0x00,0xff,0x00,
                                     0xff,0x00,0xff,0x00);
#ifdef DEBUG
  {
    vector256_t Msk;
    Msk.v = vMsk256[0];
    fprintf(stdout,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[15]),(Msk.u16[14]),(Msk.u16[13]),(Msk.u16[12]),(Msk.u16[11]),(Msk.u16[10]),(Msk.u16[9]),(Msk.u16[8]),(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
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
#ifdef DEBUG
    {
      vector256_t Msk;
      Msk.v = vMsk256[l];
      fprintf(stdout,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[15]),(Msk.u16[14]),(Msk.u16[13]),(Msk.u16[12]),(Msk.u16[11]),(Msk.u16[10]),(Msk.u16[9]),(Msk.u16[8]),(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
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

  if (!alignment_avx2__compatible_proc()) exit(1);

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes256          =  EPI64X_TYPE(_mm256_set1)(1);
  vBufferMask256    =  EPI32_TYPE(_mm256_set)(0, 0, 0, 3, 0, 0, 0, 3);
  vThreshold256     =  EPI8_TYPE(_mm256_set1)(u_threshold);
  vIndelOpenS256    =  EPI8_TYPE(_mm256_set1)(gapopen);
  vIndelExtendsS256 =  EPI8_TYPE(_mm256_set1)(gapextends);
  vMatchS256        =  EPI8_TYPE(_mm256_set1)(match);
  vMismatchS256     =  EPI8_TYPE(_mm256_set1)(mismatch);
  alignment_avx2__setlength_pair(readlength);
}


void alignment_avx2__init_quad(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_avx2__compatible_proc()) exit(1);

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes256          =  EPI64X_TYPE(_mm256_set1)(1);
  vBufferMask256    =  EPI32_TYPE(_mm256_set)(0, 3, 0, 3, 0, 3, 0, 3);
  vThreshold256     =  EPI8_TYPE(_mm256_set1)(u_threshold);
  vIndelOpenS256    =  EPI8_TYPE(_mm256_set1)(gapopen);
  vIndelExtendsS256 =  EPI8_TYPE(_mm256_set1)(gapextends);
  vMatchS256        =  EPI8_TYPE(_mm256_set1)(match);
  vMismatchS256     =  EPI8_TYPE(_mm256_set1)(mismatch);
  alignment_avx2__setlength_quad(readlength);
}


void alignment_avx2__init_octa(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_avx2__compatible_proc()) exit(1);

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

  if (!alignment_avx2__compatible_proc()) exit(1);

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
 * allows at most 3/4, 7/8 or 15/16 indels on each side.
 * @param genome is the compressed genome (first nucleotide is the lower bit of the first byte)
 * @param pos_genome gives the list of positions (in term of nucleotides) to be aligned with the read
 *        (you must substract 3/4, 7/8 or 15/16 potential indels according to the hit position)
 * @param read is the compressed read (first nucleotide is the lower bit of the first byte)
 * @return 0 if none is aligned to reach the given threshold (what should happened most of the time),
 *         or a bitmask 1<<(x) | 1<<(y) if x or y align correctly.
 */

int alignment_avx2__align_pair(unsigned char * genome,
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

  vector256_t        vector_genome_buffer __attribute__ ((aligned (16)));
  vector256_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI256_TYPE(_mm256_setzero)();

  vector256_t        vector_read_buffer   __attribute__ ((aligned (16)));
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

#ifdef DEBUG
      {
        vector256_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stdout,"[1]\t   A:%.16llx%.16llx,%.16llx%.16llx\n",(A.u64[3]),(A.u64[2]),(A.u64[1]),(A.u64[0]));
        fprintf(stdout,"[1]\t   B:%.16llx%.16llx,%.16llx%.16llx\n",(B.u64[3]),(B.u64[2]),(B.u64[1]),(B.u64[0]));
      }
#endif


      VTYPE256 vM;
      /* b) compute the matching score */
      {
        VTYPE256 vM_ab_MatchMask = EPI8_TYPE(_mm256_cmpeq)(vA,vB);
        VTYPE256 vM_add = SI256_TYPE(_mm256_and)(vM_ab_MatchMask,vMatchS256);
        VTYPE256 vM_sub = SI256_TYPE(_mm256_andnot)(vM_ab_MatchMask,vMismatchS256);

#ifdef DEBUG
        {
          vector256_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stdout,"[1]\t S_a:%.16llx%.16llx,%.16llx%.16llx\n",(S_a.u64[3]),(S_a.u64[2]),(S_a.u64[1]),(S_a.u64[0]));
          fprintf(stdout,"[1]\t S_s:%.16llx%.16llx,%.16llx%.16llx\n",(S_s.u64[3]),(S_s.u64[2]),(S_a.u64[1]),(S_a.u64[0]));
        }
#endif


#ifdef DEBUG
        {
          vector256_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stdout,"[1]\t ooM:%.16llx%.16llx,%.16llx%.16llx\n",(M_old_old.u64[3]),(M_old_old.u64[2]),(M_old_old.u64[1]),(M_old_old.u64[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm256_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm256_subs)(vM,vM_sub);
      }

#ifdef DEBUG
      {
        vector256_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stdout,"[1]\t>  M:%.16llx%.16llx,%.16llx%.16llx\n",(    M.u64[3]),(    M.u64[2]),(    M.u64[1]),(    M.u64[0]));
        fprintf(stdout,"[1]\t  oM:%.16llx%.16llx,%.16llx%.16llx\n",(M_old.u64[3]),(M_old.u64[2]),(M_old.u64[1]),(M_old.u64[0]));
        fprintf(stdout,"[1]\t  oI:%.16llx%.16llx,%.16llx%.16llx\n",(I_old.u64[3]),(I_old.u64[2]),(I_old.u64[1]),(I_old.u64[0]));
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

#ifdef DEBUG
      {
        vector256_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stdout,"[1]\t>  M:%.16llx%.16llx,%.16llx%.16llx\n",(M.u64[3]),(M.u64[2]),(M.u64[1]),(M.u64[0]));
        fprintf(stdout,"[1]\t>  I:%.16llx%.16llx,%.16llx%.16llx\n",(I.u64[3]),(I.u64[2]),(I.u64[1]),(I.u64[0]));
      }
#endif

      vM    = SI256_TYPE(_mm256_and)(vM,vMsk256[l]);
      vMMax = EPU8_TYPE(_mm256_max)(vMMax,vM);

#ifdef DEBUG
      {
        vector256_t Msk,M,Max;
        Msk.v = vMsk256[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stdout,"[1]\t>Msk:%.16llx%.16llx,%.16llx%.16llx\n",(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
        fprintf(stdout,"[1]\t>  M:%.16llx%.16llx,%.16llx%.16llx\n",(  M.u64[3]),(  M.u64[2]),(  M.u64[1]),(  M.u64[0]));
        fprintf(stdout,"[1]\t>Max:%.16llx%.16llx,%.16llx%.16llx\n",(Max.u64[3]),(Max.u64[2]),(Max.u64[1]),(Max.u64[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG
      fprintf(stdout,"[1]\t     --------------------------------,--------------------------------\n");
#endif
    } /* l */
  }
  {
    VTYPE256 vThresholdMask = EPU8_TYPE(_mm256_subs)(vMMax,vThreshold256);
    int result = 0;
    uint64_t u0 =  ((vector256_t)vThresholdMask).u64[0];
    uint64_t u1 =  ((vector256_t)vThresholdMask).u64[1];
    if ((u0 != (uint64_t) 0) || (u1 != (uint64_t) 0)) {
      result |= 1;
    }
    uint64_t u2 =  ((vector256_t)vThresholdMask).u64[2];
    uint64_t u3 =  ((vector256_t)vThresholdMask).u64[3];
    if ((u2 != (uint64_t) 0) || (u3 != (uint64_t) 0)) {
      result |= 2;
    }
#ifdef DEBUG
    fprintf(stdout,"[1]\t     ================================,================================\n");
#endif
    return result;
  }
}


int alignment_avx2__align_quad(unsigned char * genome,
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

  vector256_t        vector_genome_buffer __attribute__ ((aligned (16)));
  vector256_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI256_TYPE(_mm256_setzero)();

  vector256_t        vector_read_buffer   __attribute__ ((aligned (16)));
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

#ifdef DEBUG
      {
        vector256_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stdout,"[1]\t   A:%.16llx,%.16llx,%.16llx,%.16llx\n",(A.u64[3]),(A.u64[2]),(A.u64[1]),(A.u64[0]));
        fprintf(stdout,"[1]\t   B:%.16llx,%.16llx,%.16llx,%.16llx\n",(B.u64[3]),(B.u64[2]),(B.u64[1]),(B.u64[0]));
      }
#endif


      VTYPE256 vM;
      /* b) compute the matching score */
      {
        VTYPE256 vM_ab_MatchMask = EPI8_TYPE(_mm256_cmpeq)(vA,vB);
        VTYPE256 vM_add = SI256_TYPE(_mm256_and)(vM_ab_MatchMask,vMatchS256);
        VTYPE256 vM_sub = SI256_TYPE(_mm256_andnot)(vM_ab_MatchMask,vMismatchS256);

#ifdef DEBUG
        {
          vector256_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stdout,"[1]\t S_a:%.16llx,%.16llx,%.16llx,%.16llx\n",(S_a.u64[3]),(S_a.u64[2]),(S_a.u64[1]),(S_a.u64[0]));
          fprintf(stdout,"[1]\t S_s:%.16llx,%.16llx,%.16llx,%.16llx\n",(S_s.u64[3]),(S_s.u64[2]),(S_s.u64[1]),(S_s.u64[0]));
        }
#endif


#ifdef DEBUG
        {
          vector256_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stdout,"[1]\t ooM:%.16llx,%.16llx,%.16llx,%.16llx\n",(M_old_old.u64[3]),(M_old_old.u64[2]),(M_old_old.u64[1]),(M_old_old.u64[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm256_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm256_subs)(vM,vM_sub);
      }

#ifdef DEBUG
      {
        vector256_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stdout,"[1]\t>  M:%.16llx,%.16llx,%.16llx,%.16llx\n",(    M.u64[3]),(    M.u64[2]),(    M.u64[1]),(    M.u64[0]));
        fprintf(stdout,"[1]\t  oM:%.16llx,%.16llx,%.16llx,%.16llx\n",(M_old.u64[3]),(M_old.u64[2]),(M_old.u64[1]),(M_old.u64[0]));
        fprintf(stdout,"[1]\t  oI:%.16llx,%.16llx,%.16llx,%.16llx\n",(I_old.u64[3]),(I_old.u64[2]),(I_old.u64[1]),(I_old.u64[0]));
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

#ifdef DEBUG
      {
        vector256_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stdout,"[1]\t>  M:%.16llx,%.16llx,%.16llx,%.16llx\n",(M.u64[3]),(M.u64[2]),(M.u64[1]),(M.u64[0]));
        fprintf(stdout,"[1]\t>  I:%.16llx,%.16llx,%.16llx,%.16llx\n",(I.u64[3]),(I.u64[2]),(I.u64[1]),(I.u64[0]));
      }
#endif

      vM    = SI256_TYPE(_mm256_and)(vM,vMsk256[l]);
      vMMax = EPU8_TYPE(_mm256_max)(vMMax,vM);

#ifdef DEBUG
      {
        vector256_t Msk,M,Max;
        Msk.v = vMsk256[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stdout,"[1]\t>Msk:%.16llx,%.16llx,%.16llx,%.16llx\n",(Msk.u64[3]),(Msk.u64[2]),(Msk.u64[1]),(Msk.u64[0]));
        fprintf(stdout,"[1]\t>  M:%.16llx,%.16llx,%.16llx,%.16llx\n",(  M.u64[3]),(  M.u64[2]),(  M.u64[1]),(  M.u64[0]));
        fprintf(stdout,"[1]\t>Max:%.16llx,%.16llx,%.16llx,%.16llx\n",(Max.u64[3]),(Max.u64[2]),(Max.u64[1]),(Max.u64[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG
      fprintf(stdout,"[1]\t     ----------------,----------------,----------------,----------------\n");
#endif
    } /* l */
  }
  {
    VTYPE256 vThresholdMask = EPU8_TYPE(_mm256_subs)(vMMax,vThreshold256);
    int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE256)/(int)sizeof(uint64_t);x++) {
      uint64_t u =  ((vector256_t)vThresholdMask).u64[x];
      if (u != (uint64_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG
    fprintf(stdout,"[1]\t     ================,================,================,================\n");
#endif
    return result;
  }
}


int alignment_avx2__align_octa(unsigned char * genome,
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

  vector256_t        vector_genome_buffer __attribute__ ((aligned (16)));
  vector256_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI256_TYPE(_mm256_setzero)();

  vector256_t        vector_read_buffer   __attribute__ ((aligned (16)));
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

#ifdef DEBUG
      {
        vector256_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stdout,"[1]\t   A:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(A.u32[7]),(A.u32[6]),(A.u32[5]),(A.u32[4]),(A.u32[3]),(A.u32[2]),(A.u32[1]),(A.u32[0]));
        fprintf(stdout,"[1]\t   B:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(B.u32[7]),(B.u32[6]),(B.u32[5]),(B.u32[4]),(B.u32[3]),(B.u32[2]),(B.u32[1]),(B.u32[0]));
      }
#endif


      VTYPE256 vM;
      /* b) compute the matching score */
      {
        VTYPE256 vM_ab_MatchMask = EPI8_TYPE(_mm256_cmpeq)(vA,vB);
        VTYPE256 vM_add = SI256_TYPE(_mm256_and)(vM_ab_MatchMask,vMatchS256);
        VTYPE256 vM_sub = SI256_TYPE(_mm256_andnot)(vM_ab_MatchMask,vMismatchS256);

#ifdef DEBUG
        {
          vector256_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stdout,"[1]\t S_a:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(S_a.u32[7]),(S_a.u32[6]),(S_a.u32[5]),(S_a.u32[4]),(S_a.u32[3]),(S_a.u32[2]),(S_a.u32[1]),(S_a.u32[0]));
          fprintf(stdout,"[1]\t S_s:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(S_s.u32[7]),(S_s.u32[6]),(S_s.u32[5]),(S_s.u32[4]),(S_s.u32[3]),(S_s.u32[2]),(S_s.u32[1]),(S_s.u32[0]));
        }
#endif


#ifdef DEBUG
        {
          vector256_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stdout,"[1]\t ooM:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(M_old_old.u32[7]),(M_old_old.u32[6]),(M_old_old.u32[5]),(M_old_old.u32[4]),(M_old_old.u32[3]),(M_old_old.u32[2]),(M_old_old.u32[1]),(M_old_old.u32[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm256_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm256_subs)(vM,vM_sub);
      }

#ifdef DEBUG
      {
        vector256_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stdout,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(    M.u32[7]),(    M.u32[6]),(    M.u32[5]),(    M.u32[4]),(    M.u32[3]),(    M.u32[2]),(    M.u32[1]),(    M.u32[0]));
        fprintf(stdout,"[1]\t  oM:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(M_old.u32[7]),(M_old.u32[6]),(M_old.u32[5]),(M_old.u32[4]),(M_old.u32[3]),(M_old.u32[2]),(M_old.u32[1]),(M_old.u32[0]));
        fprintf(stdout,"[1]\t  oI:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(I_old.u32[7]),(I_old.u32[6]),(I_old.u32[5]),(I_old.u32[4]),(I_old.u32[3]),(I_old.u32[2]),(I_old.u32[1]),(I_old.u32[0]));
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

#ifdef DEBUG
      {
        vector256_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stdout,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(M.u32[7]),(M.u32[6]),(M.u32[5]),(M.u32[4]),(M.u32[3]),(M.u32[2]),(M.u32[1]),(M.u32[0]));
        fprintf(stdout,"[1]\t>  I:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(I.u32[7]),(I.u32[6]),(I.u32[5]),(I.u32[4]),(I.u32[3]),(I.u32[2]),(I.u32[1]),(I.u32[0]));
      }
#endif

      vM    = SI256_TYPE(_mm256_and)(vM,vMsk256[l]);
      vMMax = EPU8_TYPE(_mm256_max)(vMMax,vM);

#ifdef DEBUG
      {
        vector256_t Msk,M,Max;
        Msk.v = vMsk256[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stdout,"[1]\t>Msk:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[7]),(Msk.u32[6]),(Msk.u32[5]),(Msk.u32[4]),(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
        fprintf(stdout,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(  M.u32[7]),(  M.u32[6]),(  M.u32[5]),(  M.u32[4]),(  M.u32[3]),(  M.u32[2]),(  M.u32[1]),(  M.u32[0]));
        fprintf(stdout,"[1]\t>Max:%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x,%.8x\n",(Max.u32[7]),(Max.u32[6]),(Max.u32[5]),(Max.u32[4]),(Max.u32[3]),(Max.u32[2]),(Max.u32[1]),(Max.u32[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG
      fprintf(stdout,"[1]\t     --------,--------,--------,--------,--------,--------,--------,--------\n");
#endif
    } /* l */
  }
  {
    VTYPE256 vThresholdMask = EPU8_TYPE(_mm256_subs)(vMMax,vThreshold256);
    int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE256)/(int)sizeof(uint32_t);x++) {
      uint32_t u =  ((vector256_t)vThresholdMask).u32[x];
      if (u != (uint32_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG
    fprintf(stdout,"[1]\t     ========,========,========,========,========,========,========,========\n");
#endif
    return result;
  }
}


int alignment_avx2__align_hexa(unsigned char * genome,
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

  vector256_t        vector_genome_buffer __attribute__ ((aligned (16)));
  vector256_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI256_TYPE(_mm256_setzero)();

  vector256_t        vector_read_buffer   __attribute__ ((aligned (16)));
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

#ifdef DEBUG
      {
        vector256_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stdout,"[1]\t   A:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(A.u16[15]),(A.u16[14]),(A.u16[13]),(A.u16[12]),(A.u16[11]),(A.u16[10]),(A.u16[9]),(A.u16[8]),(A.u16[7]),(A.u16[6]),(A.u16[5]),(A.u16[4]),(A.u16[3]),(A.u16[2]),(A.u16[1]),(A.u16[0]));
        fprintf(stdout,"[1]\t   B:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(B.u16[15]),(B.u16[14]),(B.u16[13]),(B.u16[12]),(B.u16[11]),(B.u16[10]),(B.u16[9]),(B.u16[8]),(B.u16[7]),(B.u16[6]),(B.u16[5]),(B.u16[4]),(B.u16[3]),(B.u16[2]),(B.u16[1]),(B.u16[0]));
      }
#endif


      VTYPE256 vM;
      /* b) compute the matching score */
      {
        VTYPE256 vM_ab_MatchMask = EPI8_TYPE(_mm256_cmpeq)(vA,vB);
        VTYPE256 vM_add = SI256_TYPE(_mm256_and)(vM_ab_MatchMask,vMatchS256);
        VTYPE256 vM_sub = SI256_TYPE(_mm256_andnot)(vM_ab_MatchMask,vMismatchS256);

#ifdef DEBUG
        {
          vector256_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stdout,"[1]\t S_a:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(S_a.u16[15]),(S_a.u16[14]),(S_a.u16[13]),(S_a.u16[12]),(S_a.u16[11]),(S_a.u16[10]),(S_a.u16[9]),(S_a.u16[8]),(S_a.u16[7]),(S_a.u16[6]),(S_a.u16[5]),(S_a.u16[4]),(S_a.u16[3]),(S_a.u16[2]),(S_a.u16[1]),(S_a.u16[0]));
          fprintf(stdout,"[1]\t S_s:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(S_s.u16[15]),(S_s.u16[14]),(S_s.u16[13]),(S_s.u16[12]),(S_s.u16[11]),(S_s.u16[10]),(S_s.u16[9]),(S_s.u16[8]),(S_s.u16[7]),(S_s.u16[6]),(S_s.u16[5]),(S_s.u16[4]),(S_s.u16[3]),(S_s.u16[2]),(S_s.u16[1]),(S_s.u16[0]));
        }
#endif


#ifdef DEBUG
        {
          vector256_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stdout,"[1]\t ooM:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M_old_old.u16[15]),(M_old_old.u16[14]),(M_old_old.u16[13]),(M_old_old.u16[12]),(M_old_old.u16[11]),(M_old_old.u16[10]),(M_old_old.u16[9]),(M_old_old.u16[8]),(M_old_old.u16[7]),(M_old_old.u16[6]),(M_old_old.u16[5]),(M_old_old.u16[4]),(M_old_old.u16[3]),(M_old_old.u16[2]),(M_old_old.u16[1]),(M_old_old.u16[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm256_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm256_subs)(vM,vM_sub);
      }

#ifdef DEBUG
      {
        vector256_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stdout,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(    M.u16[15]),(    M.u16[14]),(    M.u16[13]),(    M.u16[12]),(    M.u16[11]),(    M.u16[10]),(    M.u16[9]),(    M.u16[8]),(    M.u16[7]),(    M.u16[6]),(    M.u16[5]),(    M.u16[4]),(    M.u16[3]),(    M.u16[2]),(    M.u16[1]),(    M.u16[0]));
        fprintf(stdout,"[1]\t  oM:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M_old.u16[15]),(M_old.u16[14]),(M_old.u16[13]),(M_old.u16[12]),(M_old.u16[11]),(M_old.u16[10]),(M_old.u16[9]),(M_old.u16[8]),(M_old.u16[7]),(M_old.u16[6]),(M_old.u16[5]),(M_old.u16[4]),(M_old.u16[3]),(M_old.u16[2]),(M_old.u16[1]),(M_old.u16[0]));
        fprintf(stdout,"[1]\t  oI:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(I_old.u16[15]),(I_old.u16[14]),(I_old.u16[13]),(I_old.u16[12]),(I_old.u16[11]),(I_old.u16[10]),(I_old.u16[9]),(I_old.u16[8]),(I_old.u16[7]),(I_old.u16[6]),(I_old.u16[5]),(I_old.u16[4]),(I_old.u16[3]),(I_old.u16[2]),(I_old.u16[1]),(I_old.u16[0]));
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

#ifdef DEBUG
      {
        vector256_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stdout,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M.u16[15]),(M.u16[14]),(M.u16[13]),(M.u16[12]),(M.u16[11]),(M.u16[10]),(M.u16[9]),(M.u16[8]),(M.u16[7]),(M.u16[6]),(M.u16[5]),(M.u16[4]),(M.u16[3]),(M.u16[2]),(M.u16[1]),(M.u16[0]));
        fprintf(stdout,"[1]\t>  I:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(I.u16[15]),(I.u16[14]),(I.u16[13]),(I.u16[12]),(I.u16[11]),(I.u16[10]),(I.u16[9]),(I.u16[8]),(I.u16[7]),(I.u16[6]),(I.u16[5]),(I.u16[4]),(I.u16[3]),(I.u16[2]),(I.u16[1]),(I.u16[0]));
      }
#endif

      vM    = SI256_TYPE(_mm256_and)(vM,vMsk256[l]);
      vMMax = EPU8_TYPE(_mm256_max)(vMMax,vM);

#ifdef DEBUG
      {
        vector256_t Msk,M,Max;
        Msk.v = vMsk256[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stdout,"[1]\t>Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[15]),(Msk.u16[14]),(Msk.u16[13]),(Msk.u16[12]),(Msk.u16[11]),(Msk.u16[10]),(Msk.u16[9]),(Msk.u16[8]),(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
        fprintf(stdout,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(  M.u16[15]),(  M.u16[14]),(  M.u16[13]),(  M.u16[12]),(  M.u16[11]),(  M.u16[10]),(  M.u16[9]),(  M.u16[8]),(  M.u16[7]),(  M.u16[6]),(  M.u16[5]),(  M.u16[4]),(  M.u16[3]),(  M.u16[2]),(  M.u16[1]),(  M.u16[0]));
        fprintf(stdout,"[1]\t>Max:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Max.u16[15]),(Max.u16[14]),(Max.u16[13]),(Max.u16[12]),(Max.u16[11]),(Max.u16[10]),(Max.u16[9]),(Max.u16[8]),(Max.u16[7]),(Max.u16[6]),(Max.u16[5]),(Max.u16[4]),(Max.u16[3]),(Max.u16[2]),(Max.u16[1]),(Max.u16[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG
      fprintf(stdout,"[1]\t     ----,----,----,----,----,----,----,----,----,----,----,----,----,----,----,----\n");
#endif
    } /* l */
  }
  {
    VTYPE256 vThresholdMask = EPU8_TYPE(_mm256_subs)(vMMax,vThreshold256);
    int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE256)/(int)sizeof(uint16_t);x++) {
      uint16_t u =  ((vector256_t)vThresholdMask).u16[x];
      if (u != (uint16_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG
    fprintf(stdout,"[1]\t     ====,====,====,====,====,====,====,====,====,====,====,====,====,====,====,====\n");
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
    printf("\033[31;1m");
    printf("\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    printf("\033[0m\n");
    exit(1);
  }
  vMsk128 = (void *) ((uintptr_t)(vMsk128unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk128[0] = EPI8_TYPE(_mm_set)(0xff,0x00,0x00,0x00,
                                  0x00,0x00,0x00,0x00,
                                  0x00,0x00,0x00,0x00,
                                  0x00,0x00,0x00,0x00);
#ifdef DEBUG
  {
    vector128_t Msk;
    Msk.v = vMsk128[0];
    fprintf(stdout,"[0]\t Msk:%.16llx%.16llx\n",(Msk.u64[1]),(Msk.u64[0]));
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
#ifdef DEBUG
    {
      vector128_t Msk;
      Msk.v = vMsk128[l];
      fprintf(stdout,"[0]\t Msk:%.16llx%.16llx\n",(Msk.u64[1]),(Msk.u64[0]));
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
    printf("\033[31;1m");
    printf("\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    printf("\033[0m\n");
    exit(1);
  }
  vMsk128 = (void *) ((uintptr_t)(vMsk128unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk128[0] = EPI8_TYPE(_mm_set)(0xff,0x00,0x00,0x00,
                                  0x00,0x00,0x00,0x00,
                                  0xff,0x00,0x00,0x00,
                                  0x00,0x00,0x00,0x00);
#ifdef DEBUG
  {
    vector128_t Msk;
    Msk.v = vMsk128[0];
    fprintf(stdout,"[0]\t Msk:%.16llx,%.16llx\n",(Msk.u64[1]),(Msk.u64[0]));
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
#ifdef DEBUG
    {
      vector128_t Msk;
      Msk.v = vMsk128[l];
      fprintf(stdout,"[0]\t Msk:%.16llx,%.16llx\n",(Msk.u64[1]),(Msk.u64[0]));
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
    printf("\033[31;1m");
    printf("\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    printf("\033[0m\n");
    exit(1);
  }
  vMsk128 = (void *) ((uintptr_t)(vMsk128unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk128[0] = EPI8_TYPE(_mm_set)(0xff,0x00,0x00,0x00,
                                  0xff,0x00,0x00,0x00,
                                  0xff,0x00,0x00,0x00,
                                  0xff,0x00,0x00,0x00);
#ifdef DEBUG
  {
    vector128_t Msk;
    Msk.v = vMsk128[0];
    fprintf(stdout,"[0]\t Msk:%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
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
#ifdef DEBUG
    {
      vector128_t Msk;
      Msk.v = vMsk128[l];
      fprintf(stdout,"[0]\t Msk:%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
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
    printf("\033[31;1m");
    printf("\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    printf("\033[0m\n");
    exit(1);
  }
  vMsk128 = (void *) ((uintptr_t)(vMsk128unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk128[0] = EPI8_TYPE(_mm_set)(0xff,0x00,0xff,0x00,
                                  0xff,0x00,0xff,0x00,
                                  0xff,0x00,0xff,0x00,
                                  0xff,0x00,0xff,0x00);
#ifdef DEBUG
  {
    vector128_t Msk;
    Msk.v = vMsk128[0];
    fprintf(stdout,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
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
#ifdef DEBUG
    {
      vector128_t Msk;
      Msk.v = vMsk128[l];
      fprintf(stdout,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
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

  if (!alignment_sse2__compatible_proc()) exit(1);

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes128          =  EPI32_TYPE(_mm_set1)(1); /* not used */
  vBufferMask128    =  EPI32_TYPE(_mm_set)(0, 0, 0, 3);
  vThreshold128     =  EPI8_TYPE(_mm_set1)(u_threshold);
  vIndelOpenS128    =  EPI8_TYPE(_mm_set1)(gapopen);
  vIndelExtendsS128 =  EPI8_TYPE(_mm_set1)(gapextends);
  vMatchS128        =  EPI8_TYPE(_mm_set1)(match);
  vMismatchS128     =  EPI8_TYPE(_mm_set1)(mismatch);
  alignment_sse2__setlength_mono(readlength);
}


void alignment_sse2__init_pair(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_sse2__compatible_proc()) exit(1);

  /* set maximal acceptable threshold for filtering */
  unsigned int u_threshold = threshold;
  if (u_threshold >= 255)
    u_threshold = 254;

  /* init some vectors */
  vOnes128          =  EPI32_TYPE(_mm_set1)(1);
  vBufferMask128    =  EPI32_TYPE(_mm_set)(0, 3, 0, 3);
  vThreshold128     =  EPI8_TYPE(_mm_set1)(u_threshold);
  vIndelOpenS128    =  EPI8_TYPE(_mm_set1)(gapopen);
  vIndelExtendsS128 =  EPI8_TYPE(_mm_set1)(gapextends);
  vMatchS128        =  EPI8_TYPE(_mm_set1)(match);
  vMismatchS128     =  EPI8_TYPE(_mm_set1)(mismatch);
  alignment_sse2__setlength_pair(readlength);
}


void alignment_sse2__init_quad(const unsigned int match, const unsigned int mismatch, const unsigned int gapopen, const unsigned int gapextends, const unsigned int threshold, const unsigned int readlength) {

  if (!alignment_sse2__compatible_proc()) exit(1);

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

  if (!alignment_sse2__compatible_proc()) exit(1);

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

int alignment_sse2__align_mono(unsigned char * genome,
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

  vector128_t        vector_genome_buffer __attribute__ ((aligned (16)));
  unsigned int       vector_genome_buffer_nbnuc = 0;

  vector128_t        vector_read_buffer   __attribute__ ((aligned (16)));
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

#ifdef DEBUG
      {
        vector128_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stdout,"[1]\t   A:%.16llx%.16llx\n",(A.u64[1]),(A.u64[0]));
        fprintf(stdout,"[1]\t   B:%.16llx%.16llx\n",(B.u64[1]),(B.u64[0]));
      }
#endif


      VTYPE128 vM;
      /* b) compute the matching score */
      {
        VTYPE128 vM_ab_MatchMask = EPI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE128 vM_add = SI128_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS128);
        VTYPE128 vM_sub = SI128_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS128);

#ifdef DEBUG
        {
          vector128_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stdout,"[1]\t S_a:%.16llx%.16llx\n",(S_a.u64[1]),(S_a.u64[0]));
          fprintf(stdout,"[1]\t S_s:%.16llx%.16llx\n",(S_s.u64[1]),(S_s.u64[0]));
        }
#endif


#ifdef DEBUG
        {
          vector128_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stdout,"[1]\t ooM:%.16llx%.16llx\n",(M_old_old.u64[1]),(M_old_old.u64[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG
      {
        vector128_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stdout,"[1]\t>  M:%.16llx%.16llx\n",(    M.u64[1]),(    M.u64[0]));
        fprintf(stdout,"[1]\t  oM:%.16llx%.16llx\n",(M_old.u64[1]),(M_old.u64[0]));
        fprintf(stdout,"[1]\t  oI:%.16llx%.16llx\n",(I_old.u64[1]),(I_old.u64[0]));
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

#ifdef DEBUG
      {
        vector128_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stdout,"[1]\t>  M:%.16llx%.16llx\n",(M.u64[1]),(M.u64[0]));
        fprintf(stdout,"[1]\t>  I:%.16llx%.16llx\n",(I.u64[1]),(I.u64[0]));
      }
#endif

      vM    = SI128_TYPE(_mm_and)(vM,vMsk128[l]);
      vMMax = EPU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG
      {
        vector128_t Msk,M,Max;
        Msk.v = vMsk128[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stdout,"[1]\t>Msk:%.16llx%.16llx\n",(Msk.u64[1]),(Msk.u64[0]));
        fprintf(stdout,"[1]\t>  M:%.16llx%.16llx\n",(  M.u64[1]),(  M.u64[0]));
        fprintf(stdout,"[1]\t>Max:%.16llx%.16llx\n",(Max.u64[1]),(Max.u64[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG
      fprintf(stdout,"[1]\t     --------------------------------\n");
#endif
    } /* l */
  }
  {
    VTYPE128 vThresholdMask = EPU8_TYPE(_mm_subs)(vMMax,vThreshold128);
    int result = 0;
    uint64_t u0 =  ((vector128_t)vThresholdMask).u64[0];
    uint64_t u1 =  ((vector128_t)vThresholdMask).u64[1];
    if ((u0 != (uint64_t) 0) || (u1 != (uint64_t) 0)) {
      result |= 1;
    }
#ifdef DEBUG
      fprintf(stdout,"[1]\t     ================================\n");
#endif
    return result;
  }
}


int alignment_sse2__align_pair(unsigned char * genome,
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

  vector128_t        vector_genome_buffer __attribute__ ((aligned (16)));
  vector128_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI128_TYPE(_mm_setzero)();

  vector128_t        vector_read_buffer   __attribute__ ((aligned (16)));
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

#ifdef DEBUG
      {
        vector128_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stdout,"[1]\t   A:%.16llx,%.16llx\n",(A.u64[1]),(A.u64[0]));
        fprintf(stdout,"[1]\t   B:%.16llx,%.16llx\n",(B.u64[1]),(B.u64[0]));
      }
#endif


      VTYPE128 vM;
      /* b) compute the matching score */
      {
        VTYPE128 vM_ab_MatchMask = EPI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE128 vM_add = SI128_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS128);
        VTYPE128 vM_sub = SI128_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS128);

#ifdef DEBUG
        {
          vector128_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stdout,"[1]\t S_a:%.16llx,%.16llx\n",(S_a.u64[1]),(S_a.u64[0]));
          fprintf(stdout,"[1]\t S_s:%.16llx,%.16llx\n",(S_s.u64[1]),(S_s.u64[0]));
        }
#endif


#ifdef DEBUG
        {
          vector128_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stdout,"[1]\t ooM:%.16llx,%.16llx\n",(M_old_old.u64[1]),(M_old_old.u64[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG
      {
        vector128_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stdout,"[1]\t>  M:%.16llx,%.16llx\n",(    M.u64[1]),(    M.u64[0]));
        fprintf(stdout,"[1]\t  oM:%.16llx,%.16llx\n",(M_old.u64[1]),(M_old.u64[0]));
        fprintf(stdout,"[1]\t  oI:%.16llx,%.16llx\n",(I_old.u64[1]),(I_old.u64[0]));
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

#ifdef DEBUG
      {
        vector128_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stdout,"[1]\t>  M:%.16llx,%.16llx\n",(M.u64[1]),(M.u64[0]));
        fprintf(stdout,"[1]\t>  I:%.16llx,%.16llx\n",(I.u64[1]),(I.u64[0]));
      }
#endif

      vM    = SI128_TYPE(_mm_and)(vM,vMsk128[l]);
      vMMax = EPU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG
      {
        vector128_t Msk,M,Max;
        Msk.v = vMsk128[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stdout,"[1]\t>Msk:%.16llx,%.16llx\n",(Msk.u64[1]),(Msk.u64[0]));
        fprintf(stdout,"[1]\t>  M:%.16llx,%.16llx\n",(  M.u64[1]),(  M.u64[0]));
        fprintf(stdout,"[1]\t>Max:%.16llx,%.16llx\n",(Max.u64[1]),(Max.u64[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG
      fprintf(stdout,"[1]\t     ----------------,----------------\n");
#endif
    } /* l */
  }
  {
    VTYPE128 vThresholdMask = EPU8_TYPE(_mm_subs)(vMMax,vThreshold128);
    int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE128)/(int)sizeof(uint64_t);x++) {
      uint64_t u =  ((vector128_t)vThresholdMask).u64[x];
      if (u != (uint64_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG
    fprintf(stdout,"[1]\t     ================,================\n");
#endif
    return result;
  }
}


int alignment_sse2__align_quad(unsigned char * genome,
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

  vector128_t        vector_genome_buffer __attribute__ ((aligned (16)));
  vector128_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI128_TYPE(_mm_setzero)();

  vector128_t        vector_read_buffer   __attribute__ ((aligned (16)));
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

#ifdef DEBUG
      {
        vector128_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stdout,"[1]\t   A:%.8x,%.8x,%.8x,%.8x\n",(A.u32[3]),(A.u32[2]),(A.u32[1]),(A.u32[0]));
        fprintf(stdout,"[1]\t   B:%.8x,%.8x,%.8x,%.8x\n",(B.u32[3]),(B.u32[2]),(B.u32[1]),(B.u32[0]));
      }
#endif


      VTYPE128 vM;
      /* b) compute the matching score */
      {
        VTYPE128 vM_ab_MatchMask = EPI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE128 vM_add = SI128_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS128);
        VTYPE128 vM_sub = SI128_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS128);

#ifdef DEBUG
        {
          vector128_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stdout,"[1]\t S_a:%.8x,%.8x,%.8x,%.8x\n",(S_a.u32[3]),(S_a.u32[2]),(S_a.u32[1]),(S_a.u32[0]));
          fprintf(stdout,"[1]\t S_s:%.8x,%.8x,%.8x,%.8x\n",(S_s.u32[3]),(S_s.u32[2]),(S_s.u32[1]),(S_s.u32[0]));
        }
#endif


#ifdef DEBUG
        {
          vector128_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stdout,"[1]\t ooM:%.8x,%.8x,%.8x,%.8x\n",(M_old_old.u32[3]),(M_old_old.u32[2]),(M_old_old.u32[1]),(M_old_old.u32[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG
      {
        vector128_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stdout,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x\n",(    M.u32[3]),(    M.u32[2]),(    M.u32[1]),(    M.u32[0]));
        fprintf(stdout,"[1]\t  oM:%.8x,%.8x,%.8x,%.8x\n",(M_old.u32[3]),(M_old.u32[2]),(M_old.u32[1]),(M_old.u32[0]));
        fprintf(stdout,"[1]\t  oI:%.8x,%.8x,%.8x,%.8x\n",(I_old.u32[3]),(I_old.u32[2]),(I_old.u32[1]),(I_old.u32[0]));
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

#ifdef DEBUG
      {
        vector128_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stdout,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x\n",(M.u32[3]),(M.u32[2]),(M.u32[1]),(M.u32[0]));
        fprintf(stdout,"[1]\t>  I:%.8x,%.8x,%.8x,%.8x\n",(I.u32[3]),(I.u32[2]),(I.u32[1]),(I.u32[0]));
      }
#endif

      vM    = SI128_TYPE(_mm_and)(vM,vMsk128[l]);
      vMMax = EPU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG
      {
        vector128_t Msk,M,Max;
        Msk.v = vMsk128[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stdout,"[1]\t>Msk:%.8x,%.8x,%.8x,%.8x\n",(Msk.u32[3]),(Msk.u32[2]),(Msk.u32[1]),(Msk.u32[0]));
        fprintf(stdout,"[1]\t>  M:%.8x,%.8x,%.8x,%.8x\n",(  M.u32[3]),(  M.u32[2]),(  M.u32[1]),(  M.u32[0]));
        fprintf(stdout,"[1]\t>Max:%.8x,%.8x,%.8x,%.8x\n",(Max.u32[3]),(Max.u32[2]),(Max.u32[1]),(Max.u32[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG
      fprintf(stdout,"[1]\t     --------,--------,--------,--------\n");
#endif
    } /* l */
  }
  {
    VTYPE128 vThresholdMask = EPU8_TYPE(_mm_subs)(vMMax,vThreshold128);
    int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE128)/(int)sizeof(uint32_t);x++) {
      uint32_t u =  ((vector128_t)vThresholdMask).u32[x];
      if (u != (uint32_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG
    fprintf(stdout,"[1]\t     ========,========,========,========\n");
#endif
    return result;
  }
}


int alignment_sse2__align_octa(unsigned char * genome,
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

  vector128_t        vector_genome_buffer __attribute__ ((aligned (16)));
  vector128_t        vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI128_TYPE(_mm_setzero)();

  vector128_t        vector_read_buffer   __attribute__ ((aligned (16)));
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

#ifdef DEBUG
      {
        vector128_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stdout,"[1]\t   A:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(A.u16[7]),(A.u16[6]),(A.u16[5]),(A.u16[4]),(A.u16[3]),(A.u16[2]),(A.u16[1]),(A.u16[0]));
        fprintf(stdout,"[1]\t   B:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(B.u16[7]),(B.u16[6]),(B.u16[5]),(B.u16[4]),(B.u16[3]),(B.u16[2]),(B.u16[1]),(B.u16[0]));
      }
#endif


      VTYPE128 vM;
      /* b) compute the matching score */
      {
        VTYPE128 vM_ab_MatchMask = EPI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE128 vM_add = SI128_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS128);
        VTYPE128 vM_sub = SI128_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS128);

#ifdef DEBUG
        {
          vector128_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stdout,"[1]\t S_a:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(S_a.u16[7]),(S_a.u16[6]),(S_a.u16[5]),(S_a.u16[4]),(S_a.u16[3]),(S_a.u16[2]),(S_a.u16[1]),(S_a.u16[0]));
          fprintf(stdout,"[1]\t S_s:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(S_s.u16[7]),(S_s.u16[6]),(S_s.u16[5]),(S_s.u16[4]),(S_s.u16[3]),(S_s.u16[2]),(S_s.u16[1]),(S_s.u16[0]));
        }
#endif


#ifdef DEBUG
        {
          vector128_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stdout,"[1]\t ooM:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M_old_old.u16[7]),(M_old_old.u16[6]),(M_old_old.u16[5]),(M_old_old.u16[4]),(M_old_old.u16[3]),(M_old_old.u16[2]),(M_old_old.u16[1]),(M_old_old.u16[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = EPU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = EPU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG
      {
        vector128_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stdout,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(    M.u16[7]),(    M.u16[6]),(    M.u16[5]),(    M.u16[4]),(    M.u16[3]),(    M.u16[2]),(    M.u16[1]),(    M.u16[0]));
        fprintf(stdout,"[1]\t  oM:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M_old.u16[7]),(M_old.u16[6]),(M_old.u16[5]),(M_old.u16[4]),(M_old.u16[3]),(M_old.u16[2]),(M_old.u16[1]),(M_old.u16[0]));
        fprintf(stdout,"[1]\t  oI:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(I_old.u16[7]),(I_old.u16[6]),(I_old.u16[5]),(I_old.u16[4]),(I_old.u16[3]),(I_old.u16[2]),(I_old.u16[1]),(I_old.u16[0]));
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

#ifdef DEBUG
      {
        vector128_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stdout,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(M.u16[7]),(M.u16[6]),(M.u16[5]),(M.u16[4]),(M.u16[3]),(M.u16[2]),(M.u16[1]),(M.u16[0]));
        fprintf(stdout,"[1]\t>  I:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(I.u16[7]),(I.u16[6]),(I.u16[5]),(I.u16[4]),(I.u16[3]),(I.u16[2]),(I.u16[1]),(I.u16[0]));
      }
#endif

      vM    = SI128_TYPE(_mm_and)(vM,vMsk128[l]);
      vMMax = EPU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG
      {
        vector128_t Msk,M,Max;
        Msk.v = vMsk128[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stdout,"[1]\t>Msk:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[7]),(Msk.u16[6]),(Msk.u16[5]),(Msk.u16[4]),(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
        fprintf(stdout,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(  M.u16[7]),(  M.u16[6]),(  M.u16[5]),(  M.u16[4]),(  M.u16[3]),(  M.u16[2]),(  M.u16[1]),(  M.u16[0]));
        fprintf(stdout,"[1]\t>Max:%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x,%.4x\n",(Max.u16[7]),(Max.u16[6]),(Max.u16[5]),(Max.u16[4]),(Max.u16[3]),(Max.u16[2]),(Max.u16[1]),(Max.u16[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG
      fprintf(stdout,"[1]\t     ----,----,----,----,----,----,----,----\n");
#endif
    } /* l */
  }
  {
    VTYPE128 vThresholdMask = EPU8_TYPE(_mm_subs)(vMMax,vThreshold128);
    int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE128)/(int)sizeof(uint16_t);x++) {
      uint16_t u =  ((vector128_t)vThresholdMask).u16[x];
      if (u != (uint16_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG
    fprintf(stdout,"[1]\t     ====,====,====,====,====,====,====,====\n");
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
    printf("\033[31;1m");
    printf("\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    printf("\033[0m\n");
    exit(1);
  }
  vMsk64 = (void *) ((uintptr_t)(vMsk64unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk64[0] = PI8_TYPE(_mm_set)(0xff,0x00,0x00,0x00,
                                0x00,0x00,0x00,0x00);
#ifdef DEBUG
  {
    vector64_t Msk;
    Msk.v = vMsk64[0];
    fprintf(stdout,"[0]\t Msk:%.16llx\n",(Msk.u64[0]));
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
#ifdef DEBUG
    {
      vector64_t Msk;
      Msk.v = vMsk64[l];
      fprintf(stdout,"[0]\t Msk:%.16llx\n",(Msk.u64[0]));
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
    printf("\033[31;1m");
    printf("\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    printf("\033[0m\n");
    exit(1);
  }
  vMsk64 = (void *) ((uintptr_t)(vMsk64unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk64[0] = PI8_TYPE(_mm_set)(0xff,0x00,0x00,0x00,
                                0xff,0x00,0x00,0x00);
#ifdef DEBUG
  {
    vector64_t Msk;
    Msk.v = vMsk64[0];
    fprintf(stdout,"[0]\t Msk:%.8x,%.8x\n",(Msk.u32[1]),(Msk.u32[0]));
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
#ifdef DEBUG
    {
      vector64_t Msk;
      Msk.v = vMsk64[l];
      fprintf(stdout,"[0]\t Msk:%.8x,%.8x\n",(Msk.u32[1]),(Msk.u32[0]));
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
    printf("\033[31;1m");
    printf("\nNot enough available memory.\n%s:%d\n\nExiting.\n", __FILE__, __LINE__);
    printf("\033[0m\n");
    exit(1);
  }
  vMsk64 = (void *) ((uintptr_t)(vMsk64unaligned + 15) & ~0x0f);

  /* init mask table */
  vMsk64[0] = PI8_TYPE(_mm_set)(0xff,0x00,0xff,0x00,
                                0xff,0x00,0xff,0x00);
#ifdef DEBUG
  {
    vector64_t Msk;
    Msk.v = vMsk64[0];
    fprintf(stdout,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
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
#ifdef DEBUG
    {
      vector64_t Msk;
      Msk.v = vMsk64[l];
      fprintf(stdout,"[0]\t Msk:%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
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

  if (!alignment_sse__compatible_proc()) exit(1);

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

  if (!alignment_sse__compatible_proc()) exit(1);

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

  if (!alignment_sse__compatible_proc()) exit(1);

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
 * allows at most 3/4 or 7/8 indels on each side.
 * @param genome is the compressed genome (first nucleotide is the lower bit of the first byte)
 * @param pos_genome gives the list of positions (in term of nucleotides) to be aligned with the read
 *        (you must substract 3/4 or 7/8 potential indels according to the hit position)
 * @param read is the compressed read (first nucleotide is the lower bit of the first byte)
 * @return 0 if none is aligned to reach the given threshold (what should happened most of the time),
 *         or a bitmask 1<<(x) | 1<<(y) if x or y align correctly.
 */

int alignment_sse__align_mono(unsigned char * genome,
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

  vector64_t         vector_genome_buffer __attribute__ ((aligned (16)));
  unsigned int       vector_genome_buffer_nbnuc = 0;

  vector64_t         vector_read_buffer   __attribute__ ((aligned (16)));
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

#ifdef DEBUG
      {
        vector64_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stdout,"[1]\t   A:%.16llx\n",(A.u64[0]));
        fprintf(stdout,"[1]\t   B:%.16llx\n",(B.u64[0]));
      }
#endif


      VTYPE64 vM;
      /* b) compute the matching score */
      {
        VTYPE64 vM_ab_MatchMask = PI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE64 vM_add = SI64_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS64);
        VTYPE64 vM_sub = SI64_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS64);

#ifdef DEBUG
        {
          vector64_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stdout,"[1]\t S_a:%.16llx\n",(S_a.u64[0]));
          fprintf(stdout,"[1]\t S_s:%.16llx\n",(S_s.u64[0]));
        }
#endif


#ifdef DEBUG
        {
          vector64_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stdout,"[1]\t ooM:%.16llx\n",(M_old_old.u64[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = PU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = PU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG
      {
        vector64_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stdout,"[1]\t>  M:%.16llx\n",(    M.u64[0]));
        fprintf(stdout,"[1]\t  oM:%.16llx\n",(M_old.u64[0]));
        fprintf(stdout,"[1]\t  oI:%.16llx\n",(I_old.u64[0]));
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

#ifdef DEBUG
      {
        vector64_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stdout,"[1]\t>  M:%.16llx\n",(M.u64[0]));
        fprintf(stdout,"[1]\t>  I:%.16llx\n",(I.u64[0]));
      }
#endif

      vM    = SI64_TYPE(_mm_and)(vM,vMsk64[l]);
      vMMax = PU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG
      {
        vector64_t Msk,M,Max;
        Msk.v = vMsk64[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stdout,"[1]\t>Msk:%.16llx\n",(Msk.u64[0]));
        fprintf(stdout,"[1]\t>  M:%.16llx\n",(  M.u64[0]));
        fprintf(stdout,"[1]\t>Max:%.16llx\n",(Max.u64[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG
      fprintf(stdout,"[1]\t     ----------------\n");
#endif
    } /* l */
  }
  {
    VTYPE64 vThresholdMask = PU8_TYPE(_mm_subs)(vMMax,vThreshold64);
    int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE64)/(int)sizeof(uint64_t);x++) {
      uint64_t u =  ((vector64_t)vThresholdMask).u64[x];
      if (u != (uint64_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG
    fprintf(stdout,"[1]\t     ================\n");
#endif
    _mm_empty();
    return result;
  }
}


int alignment_sse__align_pair(unsigned char * genome,
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

  vector64_t         vector_genome_buffer __attribute__ ((aligned (16)));
  vector64_t         vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI64_TYPE(_mm_setzero)();

  vector64_t         vector_read_buffer   __attribute__ ((aligned (16)));
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

#ifdef DEBUG
      {
        vector64_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stdout,"[1]\t   A:%.8x,%.8x\n",(A.u32[1]),(A.u32[0]));
        fprintf(stdout,"[1]\t   B:%.8x,%.8x\n",(B.u32[1]),(B.u32[0]));
      }
#endif


      VTYPE64 vM;
      /* b) compute the matching score */
      {
        VTYPE64 vM_ab_MatchMask = PI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE64 vM_add = SI64_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS64);
        VTYPE64 vM_sub = SI64_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS64);

#ifdef DEBUG
        {
          vector64_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stdout,"[1]\t S_a:%.8x,%.8x\n",(S_a.u32[1]),(S_a.u32[0]));
          fprintf(stdout,"[1]\t S_s:%.8x,%.8x\n",(S_s.u32[1]),(S_s.u32[0]));
        }
#endif


#ifdef DEBUG
        {
          vector64_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stdout,"[1]\t ooM:%.8x,%.8x\n",(M_old_old.u32[1]),(M_old_old.u32[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = PU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = PU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG
      {
        vector64_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stdout,"[1]\t>  M:%.8x,%.8x\n",(    M.u32[1]),(    M.u32[0]));
        fprintf(stdout,"[1]\t  oM:%.8x,%.8x\n",(M_old.u32[1]),(M_old.u32[0]));
        fprintf(stdout,"[1]\t  oI:%.8x,%.8x\n",(I_old.u32[1]),(I_old.u32[0]));
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

#ifdef DEBUG
      {
        vector64_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stdout,"[1]\t>  M:%.8x,%.8x\n",(M.u32[1]),(M.u32[0]));
        fprintf(stdout,"[1]\t>  I:%.8x,%.8x\n",(I.u32[1]),(I.u32[0]));
      }
#endif

      vM    = SI64_TYPE(_mm_and)(vM,vMsk64[l]);
      vMMax = PU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG
      {
        vector64_t Msk,M,Max;
        Msk.v = vMsk64[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stdout,"[1]\t>Msk:%.8x,%.8x\n",(Msk.u32[1]),(Msk.u32[0]));
        fprintf(stdout,"[1]\t>  M:%.8x,%.8x\n",(  M.u32[1]),(  M.u32[0]));
        fprintf(stdout,"[1]\t>Max:%.8x,%.8x\n",(Max.u32[1]),(Max.u32[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG
      fprintf(stdout,"[1]\t     --------,--------\n");
#endif
    } /* l */
  }
  {
    VTYPE64 vThresholdMask = PU8_TYPE(_mm_subs)(vMMax,vThreshold64);
    int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE64)/(int)sizeof(uint32_t);x++) {
      uint32_t u =  ((vector64_t)vThresholdMask).u32[x];
      if (u != (uint32_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG
    fprintf(stdout,"[1]\t     ========,========\n");
#endif
    _mm_empty();
    return result;
  }
}


int alignment_sse__align_quad(unsigned char * genome,
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

  vector64_t         vector_genome_buffer __attribute__ ((aligned (16)));
  vector64_t         vector_genome_buffer_nbnuc; vector_genome_buffer_nbnuc.v = SI64_TYPE(_mm_setzero)();

  vector64_t         vector_read_buffer   __attribute__ ((aligned (16)));
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

#ifdef DEBUG
      {
        vector64_t A,B;
        A.v = vA; B.v = vB;
        fprintf(stdout,"[1]\t   A:%.4x,%.4x,%.4x,%.4x\n",(A.u16[3]),(A.u16[2]),(A.u16[1]),(A.u16[0]));
        fprintf(stdout,"[1]\t   B:%.4x,%.4x,%.4x,%.4x\n",(B.u16[3]),(B.u16[2]),(A.u16[1]),(A.u16[0]));
      }
#endif


      VTYPE64 vM;
      /* b) compute the matching score */
      {
        VTYPE64 vM_ab_MatchMask = PI8_TYPE(_mm_cmpeq)(vA,vB);
        VTYPE64 vM_add = SI64_TYPE(_mm_and)(vM_ab_MatchMask,vMatchS64);
        VTYPE64 vM_sub = SI64_TYPE(_mm_andnot)(vM_ab_MatchMask,vMismatchS64);

#ifdef DEBUG
        {
          vector64_t S_a,S_s;
          S_a.v = vM_add; S_s.v = vM_sub;
          fprintf(stdout,"[1]\t S_a:%.4x,%.4x,%.4x,%.4x\n",(S_a.u16[3]),(S_a.u16[2]),(S_a.u16[1]),(S_a.u16[0]));
          fprintf(stdout,"[1]\t S_s:%.4x,%.4x,%.4x,%.4x\n",(S_s.u16[3]),(S_s.u16[2]),(S_s.u16[1]),(S_s.u16[0]));
        }
#endif


#ifdef DEBUG
        {
          vector64_t M_old_old;
          M_old_old.v = vM_old_old;
          fprintf(stdout,"[1]\t ooM:%.4x,%.4x,%.4x,%.4x\n",(M_old_old.u16[3]),(M_old_old.u16[2]),(M_old_old.u16[1]),(M_old_old.u16[0]));
        }
#endif

        /* same diagonal with M_old_old */
        vM = PU8_TYPE(_mm_adds)(vM_old_old,vM_add);
        vM = PU8_TYPE(_mm_subs)(vM,vM_sub);
      }

#ifdef DEBUG
      {
        vector64_t M,M_old,I_old;
        M.v     = vM;
        M_old.v = vM_old;
        I_old.v = vI_old;
        fprintf(stdout,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x\n",(    M.u16[3]),(    M.u16[2]),(    M.u16[1]),(    M.u16[0]));
        fprintf(stdout,"[1]\t  oM:%.4x,%.4x,%.4x,%.4x\n",(M_old.u16[3]),(M_old.u16[2]),(M_old.u16[1]),(M_old.u16[0]));
        fprintf(stdout,"[1]\t  oI:%.4x,%.4x,%.4x,%.4x\n",(I_old.u16[3]),(I_old.u16[2]),(I_old.u16[1]),(I_old.u16[0]));
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

#ifdef DEBUG
      {
        vector64_t M,I;
        M.v = vM;
        I.v = vI;
        fprintf(stdout,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x\n",(M.u16[3]),(M.u16[2]),(M.u16[1]),(M.u16[0]));
        fprintf(stdout,"[1]\t>  I:%.4x,%.4x,%.4x,%.4x\n",(I.u16[3]),(I.u16[2]),(I.u16[1]),(I.u16[0]));
      }
#endif

      vM    = SI64_TYPE(_mm_and)(vM,vMsk64[l]);
      vMMax = PU8_TYPE(_mm_max)(vMMax,vM);

#ifdef DEBUG
      {
        vector64_t Msk,M,Max;
        Msk.v = vMsk64[l];
        M.v   = vM;
        Max.v = vMMax;
        fprintf(stdout,"[1]\t>Msk:%.4x,%.4x,%.4x,%.4x\n",(Msk.u16[3]),(Msk.u16[2]),(Msk.u16[1]),(Msk.u16[0]));
        fprintf(stdout,"[1]\t>  M:%.4x,%.4x,%.4x,%.4x\n",(  M.u16[3]),(  M.u16[2]),(  M.u16[1]),(  M.u16[0]));
        fprintf(stdout,"[1]\t>Max:%.4x,%.4x,%.4x,%.4x\n",(Max.u16[3]),(Max.u16[2]),(Max.u16[1]),(Max.u16[0]));
      }
#endif

      vM_old_old = vM_old;
      vM_old     = vM;
      vI_old     = vI;

#ifdef DEBUG
      fprintf(stdout,"[1]\t     ----,----,----,----\n");
#endif
    } /* l */
  }
  {
    VTYPE64 vThresholdMask = PU8_TYPE(_mm_subs)(vMMax,vThreshold64);
    int result = 0;
    int x;
    for(x=0;x<(int)sizeof(VTYPE64)/(int)sizeof(uint16_t);x++) {
      uint16_t u =  ((vector64_t)vThresholdMask).u16[x];
      if (u != (uint16_t) 0) {
        result |= (1)<<x;
      }
    }
#ifdef DEBUG
    fprintf(stdout,"[1]\t     ====,====,====,====\n");
#endif
    _mm_empty();
    return result;
  }
}

#endif
