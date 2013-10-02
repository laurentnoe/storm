#ifndef _ALIGNMENT_SIMD_H_
#define _ALIGNMENT_SIMD_H_


/**
 * AVX2/SSE2/SSE check function : check if the cpu is OK for all that stuff ...
 * @return true if the processor is sse2/sse compatible, false otherwise ...
 */

int alignment_avx2__compatible_proc();

int alignment_sse2__compatible_proc();

int alignment_sse__compatible_proc();


/**
 * All the functions provided here need "avx2"/"sse2"/"sse" inctructions
 */


/**
 * AVX2/SSE2/SSE alignment init function : fix the scoring system and the length of the reads (must be called once before aligning)
 * @param match      inits the match score vector
 * @param mismatch   inits the mismatch penalty vector (positive value only)
 * @param gapopen    inits the gap penalty vector (positive value only)
 * @param gapextends inits the gap penalty vector (positive value only)
 * @param threshold  inits the scoring threshold (positive value only)
 * @param length     fixes the length of the reads that will be treated : this value can be changed with some functions below
 *        (but must not be changed too frequently).
 */

#ifdef __AVX2__
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
 * AVX2/SSE2/SSE alignment align function : does a banded smith-waterman of the given read against two parts of the genome;
 * allows at most 1/2, 3/4, 7/8 or 15/16 indels on each side.
 * @param genome is the compressed genome (first nucleotide is the lower bit of the first byte)
 * @param pos_genome gives the list of 8, 4, 2 or 1 positions (in term of nucleotides) to be aligned with the read
 *        (you must substract 1/2, 3/4, 7/8 or 15/16 potential indels according to the hit position)
 * @param read is the compressed read (first nucleotide is the lower bit of the first byte)
 * @return 0 if none is aligned to reach the given threshold (what should happened most of the time),
 *         or a bitmask 1<<(x) | 1<<(y) if x or y align correctly.
 */


#ifdef __AVX2__
int alignment_avx2__align_octa(unsigned char * genome,
                               int * pos_genome,
                               unsigned char * read);

int alignment_avx2__align_quad(unsigned char * genome,
                               int * pos_genome,
                               unsigned char * read);

int alignment_avx2__align_pair(unsigned char * genome,
                               int * pos_genome,
                               unsigned char * read);
#endif
#ifdef __SSE2__
int alignment_sse2__align_octa(unsigned char * genome,
                               int * pos_genome,
                               unsigned char * read);

int alignment_sse2__align_quad(unsigned char * genome,
                               int * pos_genome,
                               unsigned char * read);

int alignment_sse2__align_pair(unsigned char * genome,
                               int * pos_genome,
                               unsigned char * read);

int alignment_sse2__align_mono(unsigned char * genome,
                               int * pos_genome,
                               unsigned char * read);
#endif
#ifdef __SSE__
int alignment_sse__align_quad(unsigned char * genome,
                              int * pos_genome,
                              unsigned char * read);

int alignment_sse__align_pair(unsigned char * genome,
                              int * pos_genome,
                              unsigned char * read);

int alignment_sse__align_mono(unsigned char * genome,
                              int * pos_genome,
                              unsigned char * read);
#endif


/**
 * AVX2/SSE2/SSE alignment init read function : modify the read length when needed (but must not be changed too frequently).
 * @param readlength gives the read length (number of nucleotides inside the read)
 */

#ifdef __AVX2__
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
 * AVX2/SSE2/SSE alignment clean function : free allocated memory
 */

#ifdef __AVX2__
void alignment_avx2__clean();
#endif
#ifdef __SSE2__
void alignment_sse2__clean();
#endif
#ifdef __SSE__
void alignment_sse__clean();
#endif

#endif /* _ALIGNMENT_SIMD_H_ */
