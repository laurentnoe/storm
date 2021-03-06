#ifndef _ALIGNMENT_SIMD_H_
#define _ALIGNMENT_SIMD_H_



/**
 * AVX512BW/AVX2/SSE2/SSE/"not" : number of allowed indels relates to the SIMD filter and the possible size of the bandwidth alignment
 */
#ifdef __AVX512BW__
#define INDEL_DATA_VECTOR_SIZE (16)
#else
#ifdef __AVX2__
#define INDEL_DATA_VECTOR_SIZE (16)
#else
#ifdef __SSE2__
#define INDEL_DATA_VECTOR_SIZE (16)
#else
#ifdef __SSE__
#define INDEL_DATA_VECTOR_SIZE (8)
#else
/* No filter. Default to 7 indels (above this, the alignment algorithm would be much too slow). */
#define INDEL_DATA_VECTOR_SIZE (8)
#endif
#endif
#endif
#endif

/**
 * AVX512BW/AVX2/SSE2/SSE check function : check if the CPU is OK for all that stuff ...
 * @return true if the processor is avx512bw/avx2/sse2/sse compatible, false otherwise ...
 */
int alignment_avx512bw__compatible_proc(void);

int alignment_avx2__compatible_proc(void);

int alignment_sse2__compatible_proc(void);

int alignment_sse__compatible_proc(void);


/**
 * All the functions provided here need "avx512"/"avx2"/"sse2"/"sse" inctructions
 */


/**
 * AVX512BW/AVX2/SSE2/SSE alignment init function : fix the scoring system and the length of the reads (must be called once before aligning)
 * @param match      inits the match score vector (positive value only)
 * @param mismatch   inits the mismatch penalty vector (positive value only)
 * @param gapopen    inits the gap penalty vector (positive value only)
 * @param gapextends inits the gap penalty vector (positive value only)
 * @param threshold  inits the scoring threshold (positive value only)
 * @param length     fixes the length of the reads that will be treated : this value can be changed with some functions below
 *        (but must not be changed too frequently).
 */

#ifdef __AVX512BW__
void alignment_avx512bw__init_tria(const unsigned int match,
                                   const unsigned int mismatch,
                                   const unsigned int gapopen,
                                   const unsigned int gapextends,
                                   const unsigned int threshold,
                                   const unsigned int readlength);

void alignment_avx512bw__init_hexa(const unsigned int match,
                                   const unsigned int mismatch,
                                   const unsigned int gapopen,
                                   const unsigned int gapextends,
                                   const unsigned int threshold,
                                   const unsigned int readlength);

void alignment_avx512bw__init_octa(const unsigned int match,
                                   const unsigned int mismatch,
                                   const unsigned int gapopen,
                                   const unsigned int gapextends,
                                   const unsigned int threshold,
                                   const unsigned int readlength);

void alignment_avx512bw__init_quad(const unsigned int match,
                                   const unsigned int mismatch,
                                   const unsigned int gapopen,
                                   const unsigned int gapextends,
                                   const unsigned int threshold,
                                   const unsigned int readlength);
#endif
#ifdef __AVX2__
void alignment_avx2__init_hexa(const unsigned int match,
                               const unsigned int mismatch,
                               const unsigned int gapopen,
                               const unsigned int gapextends,
                               const unsigned int threshold,
                               const unsigned int readlength);

void alignment_avx2__init_octa(const unsigned int match,
                               const unsigned int mismatch,
                               const unsigned int gapopen,
                               const unsigned int gapextends,
                               const unsigned int threshold,
                               const unsigned int readlength);

void alignment_avx2__init_quad(const unsigned int match,
                               const unsigned int mismatch,
                               const unsigned int gapopen,
                               const unsigned int gapextends,
                               const unsigned int threshold,
                               const unsigned int readlength);

void alignment_avx2__init_pair(const unsigned int match,
                               const unsigned int mismatch,
                               const unsigned int gapopen,
                               const unsigned int gapextends,
                               const unsigned int threshold,
                               const unsigned int readlength);
#endif
#ifdef __SSE2__
void alignment_sse2__init_octa(const unsigned int match,
                               const unsigned int mismatch,
                               const unsigned int gapopen,
                               const unsigned int gapextends,
                               const unsigned int threshold,
                               const unsigned int readlength);

void alignment_sse2__init_quad(const unsigned int match,
                               const unsigned int mismatch,
                               const unsigned int gapopen,
                               const unsigned int gapextends,
                               const unsigned int threshold,
                               const unsigned int readlength);

void alignment_sse2__init_pair(const unsigned int match,
                               const unsigned int mismatch,
                               const unsigned int gapopen,
                               const unsigned int gapextends,
                               const unsigned int threshold,
                               const unsigned int readlength);

void alignment_sse2__init_mono(const unsigned int match,
                               const unsigned int mismatch,
                               const unsigned int gapopen,
                               const unsigned int gapextends,
                               const unsigned int threshold,
                               const unsigned int readlength);
#endif
#ifdef __SSE__
void alignment_sse__init_quad(const unsigned int match,
                              const unsigned int mismatch,
                              const unsigned int gapopen,
                              const unsigned int gapextends,
                              const unsigned int threshold,
                              const unsigned int readlength);

void alignment_sse__init_pair(const unsigned int match,
                              const unsigned int mismatch,
                              const unsigned int gapopen,
                              const unsigned int gapextends,
                              const unsigned int threshold,
                              const unsigned int readlength);

void alignment_sse__init_mono(const unsigned int match,
                              const unsigned int mismatch,
                              const unsigned int gapopen,
                              const unsigned int gapoextends,
                              const unsigned int threshold,
                              const unsigned int readlength);
#endif


/**
 * AVX512BW/AVX2/SSE2/SSE alignment align function : does a banded Smith-Waterman of the given read against two parts of the genome;
 * allows at most 1/2, 3/4, 7/8 or 15/16 indels on each left/right (or right/left) side.
 * @param genome is the compressed genome (first nucleotide are the two lower bits of the first byte)
 * @param pos_genome gives the list of 32, 16, 8, 4, 2 or 1 positions (in term of nucleotides) to be aligned with the read
 *        (you must substract 1 (or 2), 3 (or 4), 7 (or 8), 15 (or 16) potential indels respectively according to the hit position)
 * @param read is the compressed read (first nucleotide are the two lower bits of the first byte)
 * @return 0 if none is aligned to reach the given threshold (what should happened most of the time),
 *         or a bitmask 1<<(x) | 1<<(y) if x or y align correctly.
 */


#ifdef __AVX512BW__
unsigned int alignment_avx512bw__align_tria(unsigned char * genome,
                                            int * pos_genome,
                                            unsigned char * read);

unsigned int alignment_avx512bw__align_hexa(unsigned char * genome,
                                            int * pos_genome,
                                            unsigned char * read);

unsigned int alignment_avx512bw__align_octa(unsigned char * genome,
                                            int * pos_genome,
                                            unsigned char * read);

unsigned int alignment_avx512bw__align_quad(unsigned char * genome,
                                            int * pos_genome,
                                            unsigned char * read);
#endif
#ifdef __AVX2__
unsigned int alignment_avx2__align_hexa(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read);

unsigned int alignment_avx2__align_octa(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read);

unsigned int alignment_avx2__align_quad(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read);

unsigned int alignment_avx2__align_pair(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read);
#endif
#ifdef __SSE2__
unsigned int alignment_sse2__align_octa(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read);

unsigned int alignment_sse2__align_quad(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read);

unsigned int alignment_sse2__align_pair(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read);

unsigned int alignment_sse2__align_mono(unsigned char * genome,
                                        int * pos_genome,
                                        unsigned char * read);
#endif
#ifdef __SSE__
unsigned int alignment_sse__align_quad(unsigned char * genome,
                                       int * pos_genome,
                                       unsigned char * read);

unsigned int alignment_sse__align_pair(unsigned char * genome,
                                       int * pos_genome,
                                       unsigned char * read);

unsigned int alignment_sse__align_mono(unsigned char * genome,
                                       int * pos_genome,
                                       unsigned char * read);
#endif


/**
 * AVX512BW/AVX2/SSE2/SSE alignment init read function : modify the read length when needed (but must not be changed too frequently).
 * @param readlength gives the read length (number of nucleotides inside the read)
 */

#ifdef __AVX512BW__
void alignment_avx512bw__setlength_tria(unsigned int readlength);

void alignment_avx512bw__setlength_hexa(unsigned int readlength);

void alignment_avx512bw__setlength_octa(unsigned int readlength);

void alignment_avx512bw__setlength_quad(unsigned int readlength);
#endif
#ifdef __AVX2__
void alignment_avx2__setlength_hexa(unsigned int readlength);

void alignment_avx2__setlength_octa(unsigned int readlength);

void alignment_avx2__setlength_quad(unsigned int readlength);

void alignment_avx2__setlength_pair(unsigned int readlength);
#endif
#ifdef __SSE2__
void alignment_sse2__setlength_octa(unsigned int readlength);

void alignment_sse2__setlength_quad(unsigned int readlength);

void alignment_sse2__setlength_pair(unsigned int readlength);

void alignment_sse2__setlength_mono(unsigned int readlength);
#endif
#ifdef __SSE__
void alignment_sse__setlength_quad(unsigned int readlength);

void alignment_sse__setlength_pair(unsigned int readlength);

void alignment_sse__setlength_mono(unsigned int readlength);
#endif

/**
 * AVX512BW/AVX2/SSE2/SSE alignment clean function : free allocated memory
 */

#ifdef __AVX512BW__
void alignment_avx512bw__clean(void);
#endif
#ifdef __AVX2__
void alignment_avx2__clean(void);
#endif
#ifdef __SSE2__
void alignment_sse2__clean(void);
#endif
#ifdef __SSE__
void alignment_sse__clean(void);
#endif

#endif /* _ALIGNMENT_SIMD_H_ */
