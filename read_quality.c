#include "read_quality.h"

const short READ_QUALITY_LEVEL_UPPER_BOUNDS[] = {READ_QUALITY_LEVEL0_UB, READ_QUALITY_LEVEL1_UB, READ_QUALITY_LEVEL2_UB, READ_QUALITY_LEVEL3_UB};

inline int get_quality_level(short quality) {
  int i = 0;
  while (i < MAX_READ_QUALITY_LEVEL && READ_QUALITY_LEVEL_UPPER_BOUNDS[i] < quality) {
    ++i;
  }
  return i;
}

inline QUAL_TYPE* compress_quality_sequence(const short* quals, int len, int shift) {
  int i;
  QUAL_TYPE* result;
  shift = (shift != 0) ? 1 : 0;
  SAFE_FAILURE__ALLOC(result, COMPRESSED_Q_LEN(len), QUAL_TYPE);
  memset(result,'\0',COMPRESSED_Q_LEN(len)*sizeof(QUAL_TYPE));
  for (i = 0; i < len; ++i) {
    TO_NTH_QUAL(result, i, QUALITY_LEVEL(quals[i + shift]));
  }
  return result;
}

/**
 * Obtain the reverse quality sequence (used for RC alignments)
 */
#ifdef NUCLEOTIDES
inline void quality__reverse_compressed(const QUAL_TYPE* sequence, QUAL_TYPE* dest, int len)
#else
inline void quality__reverse_compressed(const QUAL_TYPE first, const QUAL_TYPE* sequence, QUAL_TYPE* dest, int len)
#endif
{
  int i;
  memset(dest, 0x0, COMPRESSED_Q_LEN(len));
#ifdef NUCLEOTIDES
  for (i = 0; i < len; ++i) {
    TO_NTH_QUAL(dest, i, NTH_QUAL(sequence, (len - i - 1)));
  }
#else
  TO_NTH_QUAL(dest, (len - 1), first);
  for (i = 0; i < len - 1; ++i) {
    TO_NTH_QUAL(dest, i, NTH_QUAL(sequence, (len - i - 2)));
  }
#endif
}

/* Use the read quality to decide good positions for applying the seeds */
inline void quality__build_mask(const QUAL_TYPE* sequence, int len, unsigned int* dest_mask) {
  /* Null quality : allow applying everywhere, since no info is available  */
  if (!sequence) {
    memset(dest_mask, 0xFF, COMPRESSED_QSEED_LEN(len)*sizeof(unsigned int));
    return;
  }

  /* Mask low quality parts of reads */
  memset(dest_mask, 0x00, COMPRESSED_QSEED_LEN(len)*sizeof(unsigned int));
  int i;
  for (i = 0; i < len; ++i) {
    if (NTH_QUAL(sequence, i) >= MIN_TRUSTED_QUALITY_LEVEL) {
      SET_NTH_QSEED(dest_mask, i);
    }
  }
}

inline int  quality__accept_seed(const unsigned int* qual_mask, int len, unsigned int seed_mask, int seed_len, int pos) {
  unsigned int seed_mask_1 = seed_mask;
  unsigned int seed_mask_2 = 0;
  // which element of the mask
  unsigned int i = COMPRESSED_QSEED_IDX(pos);
  unsigned int offset1 = COMPRESSED_QSEED_OFFSET(pos);
  seed_mask_1 <<= offset1;
  if (offset1 + seed_len >=  (1 << QSEED_MASK_SHIFT)) {
    seed_mask_2 = seed_mask >> ((1 << QSEED_MASK_SHIFT) - offset1);
  }
  return ((qual_mask[i] & seed_mask_1) == seed_mask_1) && ((qual_mask[i+1] & seed_mask_2) == seed_mask_2);
}

inline int  quality__accept_seed_relaxed(const unsigned int* qual_mask, int len, const SeedType* seed, int pos, double t) {
  unsigned int seed_mask_1 = seed->mask;
  unsigned int seed_mask_2 = 0;
  // which element of the mask
  unsigned int i = COMPRESSED_QSEED_IDX(pos);
  unsigned int offset1 = COMPRESSED_QSEED_OFFSET(pos);
  seed_mask_1 <<= offset1;
  if (offset1 + seed->length >= (1 << QSEED_MASK_SHIFT)) {
    seed_mask_2 = seed->mask >> ((1 << QSEED_MASK_SHIFT) - offset1);
  }

  int bit_count = 0;
  unsigned int tmp = qual_mask[i] & seed_mask_1;
  while (tmp) {
    bit_count += (tmp & 1);
    tmp >>= 1;
  }

  tmp = qual_mask[i+1] & seed_mask_2;
  while (tmp) {
    bit_count += (tmp & 1);
    tmp >>= 1;
  }
  return (bit_count * 1.0 / seed->weight >= t);
}

inline int  quality__accept_seed_relaxed2(const QUAL_TYPE* quality, int len, const SeedType* seed, int pos, int t) {
  int cummulated_qual = 0, i;
  for (i = 0; i < seed->length; ++i) {
    cummulated_qual += seed->seed[i] ? NTH_QUAL(quality, (pos+i)) : 0;
  }
  return (cummulated_qual >= t);
}
