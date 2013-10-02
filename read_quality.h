#ifndef _READ_QUALITY_H_
#define _READ_QUALITY_H_

#include "load_data.h"
#include "seed.h"

/* Read quality */
#define MAX_READ_QUALITY      93
#define READ_QUALITY_LEVELS    4
#define MAX_READ_QUALITY_LEVEL 3

#define READ_QUALITY_LEVEL0_UB  5
#define READ_QUALITY_LEVEL1_UB 10
#define READ_QUALITY_LEVEL2_UB 15
#define READ_QUALITY_LEVEL3_UB MAX_READ_QUALITY

const short READ_QUALITY_LEVEL_UPPER_BOUNDS[READ_QUALITY_LEVELS];


/*
 * Qualities
 */

typedef unsigned char QUAL_TYPE;

#define QUAL_SIZE_BITS (sizeof(QUAL_TYPE))

#define QUAL_SIZE_USED_BITS 2

#define QMASK (0x3)

#define SHIFTED_QMASK (0xC0)

#define QUAL_SEQUENCE_BLOCK_BITS (QUAL_SIZE_BITS << 3)
/* sizeof(QUAL_TYPE) * 8) */

#define QUALS_PER_SEQUENCE_BLOCK (QUAL_SIZE_BITS << 2)
/*(sizeof(QUAL_TYPE) * 8 / QUAL_SIZE_USED_BITS)*/

#define QMASK_SHIFT ((QUAL_SIZE_BITS) - (QUAL_SIZE_USED_BITS))

/* Actual allocated length for keeping len codes in a compressed sequence*/
#define COMPRESSED_Q_LEN(len) (((len) >> 2) + (((len) & 3) ? 1 : 0))
/* For QUALS_PER_SEQUENCE_BLOCK = 4:
 * ceiling(len / (float)QUALS_PER_SEQUENCE_BLOCK) = len / QUALS_PER_SEQUENCE_BLOCK + ((len % QUALS_PER_SEQUENCE_BLOCK) > 0 ? 1 : 0)
 */
#define COMPRESSED_Q_IDX(i) ((i) >> 2)
/* For QUALS_PER_SEQUENCE_BLOCK = 4: len /QUALS_PER_SEQUENCE_BLOCK) */

#define COMPRESSED_Q_OFFSET(i) ((i) & 3)
/* For QUALS_PER_SEQUENCE_BLOCK = 4: len % QUALS_PER_SEQUENCE_BLOCK) */

#define TO_NTH_QUAL(sequence, n, val)  (sequence[COMPRESSED_Q_IDX(n)] |= ((val) & QMASK) << (COMPRESSED_Q_OFFSET(n) << 1));
/* sequence[n/QUALS_PER_SEQUENCE_BLOCK] |= (((val & QMASK) << (QMASK_SHIFT)) >> ((n % QUALS_PER_SEQUENCE_BLOCK)*QUAL_SIZE_USED_BITS)); */


#define NTH_QUAL(sequence, n)   ((sequence[COMPRESSED_Q_IDX(n)] & (QMASK << (COMPRESSED_Q_OFFSET(n) << 1))) >> (COMPRESSED_Q_OFFSET(n) << 1))
/* For QUALS_PER_SEQUENCE_BLOCK = 4, QMASK = 3, QMASK_SHIFT = 6, QUAL_SIZE_USED_BITS = 2:
 * ((((QMASK << (QMASK_SHIFT)) >> ((n % QUALS_PER_SEQUENCE_BLOCK)*QUAL_SIZE_USED_BITS)) & sequence[n/QUALS_PER_SEQUENCE_BLOCK]) >> (QMASK_SHIFT - ((n%QUALS_PER_SEQUENCE_BLOCK)*QUAL_SIZE_USED_BITS)))
 */

/*
 * Seed quality mask
 */
#define QSEED_MASK  (0x1f)     /* 0x1f = 31 to detect value that are multiple of 32 */
#define QSEED_MASK_SHIFT (0x5) /* 2^5  = 32 to get true coordinates */

#define COMPRESSED_QSEED_LEN(len)  (((len) >> QSEED_MASK_SHIFT) + (((len) & QSEED_MASK) ? 1 : 0))
#define COMPRESSED_QSEED_IDX(i)    ((i) >> QSEED_MASK_SHIFT)
#define COMPRESSED_QSEED_OFFSET(i) ((i) & QSEED_MASK)
#define SET_NTH_QSEED(sequence, n)  (sequence[COMPRESSED_QSEED_IDX(n)] |= (1) << (COMPRESSED_QSEED_OFFSET(n)))
#define GET_NTH_QSEED(sequence, n)  ((sequence[COMPRESSED_QSEED_IDX(n)] & (QSEED_MASK << COMPRESSED_QSEED_OFFSET(n))) >> (COMPRESSED_QSEED_OFFSET(n)))


/* Read quality handling */
/*(READ_QUALITY_LEVELS - 1) * MIN (Q, 40) / 40 => index of the score */
/* @Deprecated */
/* #define QUALITY_LEVEL(q) ((int) ( ( (READ_QUALITY_LEVELS - 1) * MIN((q), MAX_READ_QUALITY)) / MAX_READ_QUALITY  + .5))*/
int get_quality_level(short quality);

#define QUALITY_LEVEL(q) (get_quality_level(q))

QUAL_TYPE* compress_quality_sequence(const short* qual, int len, int shift);

/**
 * Obtain the reverse quality sequence (used for RC alignments)
 */
#ifdef NUCLEOTIDES
void quality__reverse_compressed(const QUAL_TYPE* sequence, QUAL_TYPE* dest, int len);
#else
void quality__reverse_compressed(const QUAL_TYPE first, const QUAL_TYPE* sequence, QUAL_TYPE* dest, int len);
#endif

/* Use the read quality to decide good positions for applying the seeds */
#define MIN_TRUSTED_QUALITY_LEVEL 2
#define MIN_TRUSTED_CODES 0.8

/**
 * Seed masking when the quality is low
 */
// @{
void quality__build_mask(const QUAL_TYPE* sequence, int len, unsigned int* dest_mask);

int  quality__accept_seed(const unsigned int* qual_mask, int len, unsigned int seed_mask, int seed_len, int pos);

int  quality__accept_seed_relaxed(const unsigned int* qual_mask, int len, const SeedType* seed, int pos, double t);

int  quality__accept_seed_relaxed2(const QUAL_TYPE* quality, int len, const SeedType* seed, int pos, int t);
// @}
#endif /* _READ_QUALITY_H_ */
