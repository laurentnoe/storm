#include "read_quality.h"

/**
 * the default quality scheme is "sanger/illumina >=1.8/solid (color)" compatible : '!' is 33
 */
int read_quality_min_symbol_code = 33;

const short READ_QUALITY_LEVEL_UPPER_BOUNDS[] = {READ_QUALITY_LEVEL0_UB, READ_QUALITY_LEVEL1_UB, READ_QUALITY_LEVEL2_UB, READ_QUALITY_LEVEL3_UB};

inline int get_quality_level(short quality) {
  int i = 0;
  while (i < MAX_READ_QUALITY_LEVEL && READ_QUALITY_LEVEL_UPPER_BOUNDS[i] < quality) {
    ++i;
  }
  return i;
}

inline int fastq_parse_quality_from_char(char c) {
  int value = (int)c - read_quality_min_symbol_code;
  if (value < 0) {
    ERROR__("Quality level char \"%c\" {x:%2x;dec:%3d} (from base \"%c\" {x:%2x;dec:%3d}) induces a negative value \"%d\" .\nUnable to set quality value.\n", c, (int)c, (int)c, (char)read_quality_min_symbol_code, read_quality_min_symbol_code, read_quality_min_symbol_code, value);
    exit (RETURN_INPUT_ERR);
  } else {
    if (value > MAX_READ_QUALITY) {
      value = MAX_READ_QUALITY;
    }
  }
  return value;
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
