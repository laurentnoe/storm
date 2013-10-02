#include "codes.h"

#ifndef NUCLEOTIDES
const char COLOR_CODE_LETTER[COLOR_CODE_COUNT] = {'0', '1', '2', '3'};
#endif
const char BASE_CODE_LETTER[BASE_CODE_COUNT]   = {'A', 'C', 'G', 'T'};

/*
 * block alignment in bytes (usefull for alignment_simd.h that access data by block and on byte per byte)
 */
int N_BYTES = 8;

/*
 * Sequence item access
 */
#define CODE_OFFSET(offset) (offset)

#define NEW_OFFSET(offset, len) (((offset) + (len)) & 3)
/*((offset + len) % CODES_PER_SEQUENCE_BLOCK, CODES_PER_SEQUENCE_BLOCK = 4)*/

#define STRIP2(code) ((code) &= MASK)

/**
 * Convert a base name to its code.
 */
int base_to_code (const char base) {
  switch (base) {
  case 'A':
    return 0;
    break;
  case 'C':
    return 1;
    break;
  case 'G':
    return 2;
    break;
  case 'T':
    return 3;
    break;
  }
  return 0;
}

/**
 * Obtains a readable string from a compressed base sequence.
 * @param sequence
 * @param len The number of codes in the sequence (not the length of the vector!)
 * @return The reference of a string, of length len, that is the representation of the given compressed sequence.
 */
char* compressed_base_sequence_to_string(const CODE_TYPE* sequence, const int len) {
  int i;
  char* result;
  SAFE_FAILURE__ALLOC(result, len+1, char);
  for (i = 0; i < len; ++i) {
    result[i] = BASE_CODE_LETTER[(int)NTH_CODE(sequence, i)];
  }
  result[len] = '\0';
  return result;
}

/**
 * Obtains a readable string from a base sequence.
 * @param sequence
 * @param len The number of codes in the sequence
 * @return The reference of a string, of length len, that is the representation of the given sequence.
 */
char* base_sequence_to_string(const CODE_TYPE* sequence, const int len) {
  int i;
  char* result;
  SAFE_FAILURE__ALLOC(result, len+1, char);
  for (i = 0; i < len; ++i) {
    result[i] = BASE_CODE_LETTER[(int)sequence[i]];
  }
  result[len] = '\0';
  return result;
}

/**
 * Obtains a readable string from a compressed color sequence.
 * @param sequence
 * @param offset
 * @param len The number of codes in the sequence (not the length of the vector!)
 * @return The reference of a string, of length len, that is the representation of the given compressed sequence.
 */
char* compressed_code_sequence_to_string(const CODE_TYPE* sequence, const int offset, const int len) {
  int i;
  char* result;
  SAFE_FAILURE__ALLOC(result, len+1, char);
  for (i = 0; i < len; ++i) {
#ifdef NUCLEOTIDES
    result[i] = BASE_CODE_LETTER[(int)NTH_CODE(sequence, (i + CODE_OFFSET(offset)))];
#else
    result[i] = COLOR_CODE_LETTER[(int)NTH_CODE(sequence, (i + CODE_OFFSET(offset)))];
#endif
  }
  result[len] = '\0';
  return result;
}

/**
 * Obtains a readable string from a color sequence.
 * @param sequence
 * @param len The number of codes in the sequence
 * @return The reference of a string, of length len, that is the representation of the given sequence.
 */
char* code_sequence_to_string(const CODE_TYPE* sequence, const int len) {
  int i;
  char* result;
  SAFE_FAILURE__ALLOC(result, len+1, char);
  for (i = 0; i < len; ++i) {
#ifdef NUCLEOTIDES
    result[i] = BASE_CODE_LETTER[(int)sequence[i]];
#else
    result[i] = COLOR_CODE_LETTER[(int)sequence[i]];
#endif
  }
  result[len] = '\0';
  return result;
}

/**
 * Given a (readable) character sequence, this method changes it into a real code sequence by 0-ing all but the last two bits.
 * @param sequence
 * @param len
 */
void strip_code_sequence(CODE_TYPE* sequence, const int len) {
  int i;
  for (i = 0; i < len; ++i) {
    STRIP2(sequence[i]);
  }
}

/**
 * Obtains a code sequence from a string, and places it at the indicated address
 * @param src
 * @param dest
 * @param len The number of codes to be recovered from the source
 */
inline void string_to_dest_code(const char* src, CODE_TYPE* dest, const int len) {
  int i;
  for (i = 0; i < len; ++i) {
    dest[i] = STRIP(src[i]);
  }
}

/**
 * Obtains a code sequence from a string
 * @param string
 * @return The address of the created code sequence
 */
CODE_TYPE* string_to_code(const char* string) {
  int len = strlen(string);
  CODE_TYPE* sequence;
  SAFE_FAILURE__ALLOC(sequence, len, CODE_TYPE);
  string_to_dest_code(string, sequence, len);
  return sequence;
}

/**
 * Obtains a compressed code sequence from a string, and places it at the indicated address
 * @param src
 * @param dest
 * @param len The number of codes to be recovered from the source
 * @param offset The offset (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The new offset
 */
inline int string_to_dest_compressed_code(const char* src, CODE_TYPE* dest, const int len, const int offset) {
  int i;

  for (i = 0; i < len; ++i) {
#ifdef NUCLEOTIDES
    TO_NTH_CODE(dest, (i + CODE_OFFSET(offset)), STRIP(base_to_code(src[i])));
#else
    TO_NTH_CODE(dest, (i + CODE_OFFSET(offset)), STRIP(src[i]));
#endif
  }

  return NEW_OFFSET(len, offset);
}

/**
 * Obtains a compressed code sequence from a string
 * @param string
 * @return The address of the created code sequence
 */
CODE_TYPE* string_to_compressed_code(const char* string) {
  int len  = strlen(string);
  int clen = COMPRESSED_LEN_N_BYTES(len,N_BYTES);
  CODE_TYPE* result;
  SAFE_FAILURE__ALLOC(result, clen, CODE_TYPE);
  memset(result, '\0', clen);
  string_to_dest_compressed_code(string, result, len, 0);
  return result;
}

/**
 * Obtains the color code sequence from an input base sequence
 * @param src
 * @param dest
 * @param len The number of bases to consider from the source. The number of obtained colors will be len - 1.
 */
inline void base_string_to_dest_code(const char* src, CODE_TYPE* dest, const int len) {
  int i;
#ifdef NUCLEOTIDES
  for (i = 0; i < len; ++i) {
       dest[i] = BASE_SYMBOL_TO_CODE(src[i]);
  }
#else
  for (i = 0; i < len-1; ++i) {
    dest[i] = COLOR(BASE_SYMBOL_TO_CODE(src[i]), BASE_SYMBOL_TO_CODE(src[i+1]));
  }
#endif
}

/**
 * Obtains the color code sequence from an input base sequence
 * @param string
 * @return The address of the created code sequence, whose length will be 1 less than the source's length
 */

CODE_TYPE* base_string_to_code(const char* string) {
  int len = strlen(string);
  CODE_TYPE* sequence;
  SAFE_FAILURE__ALLOC(sequence, len, CODE_TYPE);
  base_string_to_dest_code(string, sequence, len);
  return sequence;
}

#ifndef NUCLEOTIDES

/**
 * Obtains the color code sequence from an input base sequence
 * @param src
 * @param dest
 * @param len The number of bases to consider from the source. The number of obtained colors will be len - 1.
 * @param offset The offset (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The new offset
 */
inline int base_string_to_dest_compressed_color_code(const char* src, CODE_TYPE* dest, const int len, const int offset) {
  int i;
  for (i = 0; i < len - 1; ++i) {
    TO_NTH_CODE(dest, (i + CODE_OFFSET(offset)), COLOR(BASE_SYMBOL_TO_CODE(src[i]), BASE_SYMBOL_TO_CODE(src[i+1])));
  }
  return NEW_OFFSET(len, offset);
}

/**
 * Obtains the color code sequence from an input base sequence
 * @param string
 * @return The address of the created code sequence, whose length will be 1 less than the source's length
 */
CODE_TYPE* base_string_to_compressed_color_code(const char* string) {
  int len  = strlen(string);
  int clen = COMPRESSED_LEN_N_BYTES(len-1,N_BYTES);
  CODE_TYPE* sequence;
  SAFE_FAILURE__ALLOC(sequence, clen, CODE_TYPE);
  memset(sequence, '\0', clen);
  base_string_to_dest_compressed_color_code(string, sequence, len, 0);
  return sequence;
}

#endif

/**
 * Obtains a compressed code sequence from an uncompressed one, and places at the indicated destination adress
 * @param src
 * @param dest
 * @param len The number of codes in the sequence
 * @param offset The offset (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The new offset
 */
inline int compress_to(const CODE_TYPE* src, CODE_TYPE* dest, const int len, const int offset) {
  int i;
  for (i = 0; i < len; ++i) {
    TO_NTH_CODE(dest, (i + CODE_OFFSET(offset)), src[i]);
  }
  return NEW_OFFSET(len, offset);
}

/**
 * Obtains a compressed code sequence from an uncompressed one
 * @param sequence
 * @param len The number of codes in the sequence
 * @return The address of the created code sequence
 */
CODE_TYPE* compress(const CODE_TYPE* sequence, const int len) {
  CODE_TYPE* result;
  int clen = COMPRESSED_LEN_N_BYTES(len,N_BYTES);
  SAFE_FAILURE__ALLOC(result, clen, CODE_TYPE);
  memset(result, '\0', clen);
  compress_to(sequence, result, len, 0);
  return result;
}

/**
 * Decompresses a compressed code sequence, and places at the indicated destination adress
 * @param src
 * @param dest
 * @param len The number of codes in the sequence (not the length of the vector!)
 * @param offset The offset from the compressed sequence's address (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The new offset
 */
inline int decompress_to(const CODE_TYPE* src, CODE_TYPE* dest, const int len, const int offset) {
  int i;
  for (i = 0; i < len; ++i) {
    dest[i] = (CODE_TYPE)NTH_CODE(src, (i + CODE_OFFSET(offset)));
  }
  return NEW_OFFSET(len, offset);
}

/**
 * Decompresses a compressed code sequence
 * @param sequence
 * @param len The number of codes in the sequence (not the length of the vector!)
 * @param offset The offset from the compressed sequence's address (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The address of the created code sequence
 */
CODE_TYPE* decompress(const CODE_TYPE* sequence, const int len, const int offset) {
  CODE_TYPE* result;
  SAFE_FAILURE__ALLOC(result, len, CODE_TYPE);
  decompress_to(sequence, result, len, offset);
  return result;
}

#ifndef NUCLEOTIDES

/**
 * Compose a sequence of colors
 * @param color_sequence The address of the sequence
 * @param len The number of colors to compose
 * @return The color resulted from the operation
 */

CODE_TYPE compose(const CODE_TYPE* color_sequence, const int len) {
  CODE_TYPE result = CODE_IDENTITY;
  int i;
  for (i = 0; i < len; ++i) {
    result = COMPOSE(result, color_sequence[i]);
  }
  return result;
}

/**
 * Compose a compressed sequence of colors
 * @param color_sequence The address of the sequence
 * @param len The number of colors to compose
 * @param offset The offset from the compressed sequence's address (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The color resulted from the operation
 */
CODE_TYPE compose_compressed(const CODE_TYPE* color_sequence, const int len, const int offset) {
  CODE_TYPE result = CODE_IDENTITY;
  int i;
  for (i = 0; i < len; ++i) {
    result = COMPOSE(result, NTH_CODE(color_sequence, (i + CODE_OFFSET(offset))));
  }
  return result;
}

/**
 * Finds the base at position len from a DNA sequence, knowing the first base in the sequence and the sequence of colors up to the wanted base
 * @param start The start base code
 * @param color_sequence The color sequence between the two bases
 * @param len The position of the wanted base code, relative to start which is considered at position 0
 * @return The code of the base at position pos
 */
inline CODE_TYPE base_color_transformation(const CODE_TYPE start, const CODE_TYPE* color_sequence, const int pos) {
  return TRANSFORM(start, compose(color_sequence, pos));
}

/**
 * Finds the base at position len from a DNA sequence, knowing the first base in the sequence and the compressed sequence of colors up to the wanted base
 * @param start The start base code
 * @param color_sequence The color sequence between the two bases
 * @param len The position of the wanted base code, relative to start which is considered at position 0
 * @param offset The offset from the compressed sequence's address (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The code of the base at position pos
 */
inline CODE_TYPE base_compressed_color_transformation(const CODE_TYPE start, const CODE_TYPE* color_sequence, const int pos, const int offset) {
  return TRANSFORM(start, compose_compressed(color_sequence, pos, offset));
}

#endif

/**
 * Displays an (uncompressed) code sequence
 * @param sequence
 * @param len The number of codes in the sequence (not the length of the vector!)
 */
void display_code_sequence(const CODE_TYPE* sequence, const int len, FILE* f) {
  int i;
  for (i = 0; i < len; ++i) {
    fprintf(f, "%d", sequence[i]);
  }
  fprintf(f, "\n");
}

/**
 * Displays a compressed code sequence
 * @param sequence
 * @param len The number of codes in the sequence (not the length of the vector!)
  */
void display_compressed_code_sequence(const CODE_TYPE* sequence, const int len, FILE* f) {
  int i;
  for (i = 0; i < len; ++i) {
    fprintf(f, "%d", NTH_CODE(sequence, i));
  }
  fprintf(f, "\n");
}
/**
 * Obtain the reverse complementary of a sequence (which, in the color code, is simply the reversed color sequence)
 */
inline void sequence__reverse(const CODE_TYPE* sequence, CODE_TYPE* dest, const int len) {
  int i;
  for (i = 0; i < len; ++i) {
#ifdef NUCLEOTIDES
    dest[len - i - 1] = BASE_CODE_COMPLEMENTARY(sequence[i]);
#else
    dest[len - i - 1] = sequence[i];
#endif
  }
}

/**
 * Obtain the reverse complementary of a sequence (which, in the color code, is simply the reversed color sequence)
 */
inline void sequence__reverse_compressed(const CODE_TYPE* sequence, CODE_TYPE* dest, const int len) {
  int i;
  memset(dest, 0x0, COMPRESSED_LEN(len));
  for (i = 0; i < len; ++i) {
#ifdef NUCLEOTIDES
    TO_NTH_CODE(dest, (len - i - 1), BASE_CODE_COMPLEMENTARY(NTH_CODE(sequence, i)));
#else
    TO_NTH_CODE(dest, (len - i - 1), NTH_CODE(sequence, i));
#endif
  }
}

