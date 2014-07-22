#ifndef _CODES_H_
#define _CODES_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

typedef unsigned char  CODE_TYPE;
typedef unsigned char  CODE_SEQUENCE_BLOCK_TYPE;

/*
 * Base codes
 */
typedef CODE_TYPE BASE_CODE_TYPE;

#define BASE_A 0x0
#define BASE_C 0x1
#define BASE_G 0x2
#define BASE_T 0x3


#ifndef NUCLEOTIDES
#define COLOR_CODE_COUNT 4
extern const char COLOR_CODE_LETTER[COLOR_CODE_COUNT];
#endif

#define BASE_CODE_COUNT 4
extern const char BASE_CODE_LETTER[BASE_CODE_COUNT];

/*
 * 'A': 01000001
 * 'C': 01000
 * 'G': 0100        1
 * 'T': 01010100
 * Just a hack...
 */
#define BASE_SYMBOL_TO_CODE(symbol) ( ((symbol & 0x4) >> 1 ) | (((symbol & 0x4) >> 2) ^ ((symbol & 0x2) >> 1)))

#define BASE_COMPLEMENTARY(symbol) (symbol == 'A' ? 'T' : (symbol == 'C' ? 'G' : (symbol == 'G' ? 'C' : (symbol == 'T' ? 'A' : symbol))))

#define BASE_CODE_COMPLEMENTARY(code) ((~code) & 0x3)

#define REVERSE_COMPLEMENT(seq) {                                                   \
    char __tmp; int __p, __seq_len = strlen(seq), __seq_half = (__seq_len + 1) / 2; \
    for (__p = 0; __p < __seq_half; ++__p) {                                        \
      __tmp = BASE_COMPLEMENTARY(seq[__p]);                                         \
      seq[__p] = BASE_COMPLEMENTARY(seq[__seq_len - __p - 1]);                      \
      seq[__seq_len - __p - 1] = __tmp;                                             \
    }                                                                               \
  }


#define IS_VALID_BASE_SYMBOL(symbol) ( symbol == BASE_CODE_LETTER[BASE_A] || symbol == BASE_CODE_LETTER[BASE_C] || symbol == BASE_CODE_LETTER[BASE_T] || symbol == BASE_CODE_LETTER[BASE_G] )

/*
 * Color codes
 */

#ifndef NUCLEOTIDES

#define COMPOSE_SAFE(color1, color2) (((color1) ^ (color2)) & 0x3)
#define TRANSFORM(base, color)       ((base) ^ (color))

typedef CODE_TYPE COLOR_CODE_TYPE;

#define CODE_BLUE   0x0
#define CODE_GREEN  0x1
#define CODE_YELLOW 0x2
#define CODE_RED    0x3

#define CODE_IDENTITY      CODE_BLUE
#define CODE_TRANSITION    CODE_YELLOW
#define CODE_COMPLEMENTARY CODE_RED
#define CODE_TRANSVERSION  CODE_GREEN

extern const char COLOR_CODE[COLOR_CODE_COUNT];
extern const char COLOR_CODE_LETTER[COLOR_CODE_COUNT];

/*
 * Base/color and Color/color relations
 */

#define COMPOSE(color1, color2)      ((color1) ^ (color2))
#define COMPOSE_SAFE(color1, color2) (((color1) ^ (color2)) & 0x3)
#define COLOR(base1, base2)          ((base1) ^ (base2))
#define TRANSFORM(base, color)       ((base) ^ (color))

#else
#define CODE_IDENTITY 0x0
#endif

#ifdef NUCLEOTIDES
  #define CHARDISPLAY(code)            (BASE_CODE_LETTER[(code)&0x3])
#else
  #define CHARDISPLAY(code)            (COLOR_CODE_LETTER[(code)&0x3])
#endif




#define CODE_SIZE_BITS (sizeof(CODE_TYPE) << 3)
/* sizeof(CODE_TYPE) * 8) */

#define CODE_SIZE_USED_BITS 2

#define MASK (0x3)
#define SHIFTED_MASK (0xC0)
#define STRIP(code)  (code & MASK)
#define CODE_SEQUENCE_BLOCK_BITS (sizeof(CODE_TYPE) << 3)
#define CODES_PER_SEQUENCE_BLOCK (sizeof(CODE_TYPE) << 2)

/* Actual allocated length for keeping len codes in a compressed sequence*/
#define COMPRESSED_LEN(len) (((len) >> 2) + (((len) & 3) ? 1 : 0))
/* Same + N bytes added to enable large block reading without "valgrind weeping" */
extern int N_BYTES;
#define COMPRESSED_LEN_N_BYTES(len,N) ((COMPRESSED_LEN(len) + ((N)-1)) & (~((N)-1)))
/* For CODES_PER_SEQUENCE_BLOCK = 4:
 * ceiling(len / (float)CODES_PER_SEQUENCE_BLOCK) = len / CODES_PER_SEQUENCE_BLOCK + ((len % CODES_PER_SEQUENCE_BLOCK) > 0 ? 1 : 0)
 */
#define COMPRESSED_IDX(i) ((i) >> 2)
/* For CODES_PER_SEQUENCE_BLOCK = 4: len /CODES_PER_SEQUENCE_BLOCK) */

#define COMPRESSED_OFFSET(i) ((i) & 3)
/* For CODES_PER_SEQUENCE_BLOCK = 4: len % CODES_PER_SEQUENCE_BLOCK) */

/* the order of the codes in the byte is 3 2 1 0, making it easier to obtain each code */
#define TO_NTH_CODE(sequence, n, code)  (sequence[COMPRESSED_IDX(n)] |= ((code) & MASK) << (COMPRESSED_OFFSET(n) << 1));
#define NTH_CODE(sequence, n)  ((sequence[COMPRESSED_IDX(n)] & (MASK << (COMPRESSED_OFFSET(n) << 1))) >> (COMPRESSED_OFFSET(n) << 1))

#define NTH_CODE_IS_POSSIBLY_NOTZERO(sequence, n) ((sequence[COMPRESSED_IDX(n)]))

/**
 * Obtains a readable string from a compressed base sequence.
 * @param sequence
 * @param len The number of codes in the sequence (not the length of the vector!)
 * @return The reference of a string, of length len, that is the representation of the given compressed sequence.
 */
char* compressed_base_sequence_to_string(const CODE_TYPE* sequence, const int len);

/**
 * Obtains a readable string from a base sequence.
 * @param sequence
 * @param len The number of codes in the sequence
 * @return The reference of a string, of length len, that is the representation of the given sequence.
 */
char* base_sequence_to_string(const CODE_TYPE* sequence, const int len);

/**
 * Obtains a readable string from a compressed code (color/base) sequence.
 * @param sequence
 * @param offset
 * @param len The number of codes in the sequence (not the length of the vector!)
 * @return The reference of a string, of length len, that is the representation of the given compressed sequence.
 */
char* compressed_code_sequence_to_string(const CODE_TYPE* sequence, const int offset, const int len);

/**
 * Obtains a readable string from a code (color/base) sequence.
 * @param sequence
 * @param len The number of codes in the sequence
 * @return The reference of a string, of length len, that is the representation of the given sequence.
 */
char* code_sequence_to_string(const CODE_TYPE* sequence, const int len);

/**
 * Given a (displayable) character sequence, this method changes it into a real code sequence by 0-ing all but the last two bits.
 * @param sequence
 * @param len
 */
void strip_code_sequence(CODE_TYPE* sequence, const int len);

/**
 * Obtains a code sequence from a string
 * @param string
 * @return The address of the created code sequence
 */
CODE_TYPE* string_to_code(const char* string);

/**
 * Obtains a code sequence from a string, and places it at the indicated address
 * @param src
 * @param dest
 * @param len The number of codes to be recovered from the source
 */
void string_to_dest_code(const char* src, CODE_TYPE* dest, const int len);

/**
 * Obtains a compressed code sequence from a string
 * @param string
 * @return The address of the created code sequence
 */
CODE_TYPE* string_to_compressed_code(const char* string);

/**
 * Obtains a compressed code sequence from a string, and places it at the indicated address
 * @param src
 * @param dest
 * @param len The number of codes to be recovered from the source
 * @param offset The offset (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The new offset
 */
int string_to_dest_compressed_code(const char* src, CODE_TYPE* dest, const int len, const int offset);

/**
 * Obtains the color code sequence from an input base sequence
 * @param string
 * @return The address of the created code sequence, whose length will be 1 less than the source's length
 */
CODE_TYPE* base_string_to_code(const char* string);

/**
 * Obtains the color code sequence from an input base sequence
 * @param src
 * @param dest
 * @param len The number of bases to consider from the source. The number of obtained colors will be len - 1.
 */
void base_string_to_dest_code(const char* src, CODE_TYPE* dest, const int len);

/**
 * Obtains the color code sequence from an input base sequence
 * @param string
 * @return The address of the created code sequence, whose length will be 1 less than the source's length
 */
CODE_TYPE* base_string_to_compressed_color_code(const char* string);

/**
 * Obtains the color code sequence from an input base sequence
 * @param src
 * @param dest
 * @param len The number of bases to consider from the source. The number of obtained colors will be len - 1.
 * @param offset The offset (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The new offset
 */
int base_string_to_dest_compressed_code(const char* src, CODE_TYPE* dest, const int len, const int offset);

/**
 * Obtains a compressed code sequence from an uncompressed one
 * @param sequence
 * @param len The number of codes in the sequence
 * @return The address of the created code sequence
 */
CODE_TYPE* compress(const CODE_TYPE* sequence, const int len);

/**
 * Obtains a compressed code sequence from an uncompressed one, and places at the indicated destination adress
 * @param src
 * @param dest
 * @param len The number of codes in the sequence
 * @param offset The offset (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The new offset
 */
int compress_to(const CODE_TYPE* src, CODE_TYPE* dest, const int len, const int offset);

/**
 * Decompresses a compressed code sequence
 * @param sequence
 * @param len The number of codes in the sequence (not the length of the vector!)
 * @param offset The offset from the compressed sequence's address (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The address of the created code sequence
 */
CODE_TYPE* decompress(const CODE_TYPE* sequence, const int len, const int offset);

/**
 * Decompresses a compressed code sequence, and places at the indicated destination adress
 * @param src
 * @param dest
 * @param len The number of codes in the sequence (not the length of the vector!)
 * @param offset The offset from the compressed sequence's address (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The new offset
 */
int decompress_to(const CODE_TYPE* src, CODE_TYPE* dest, const int len, const int offset);


#ifndef NUCLEOTIDES
/**
 * Compose a sequence of colors
 * @param sequence The address of the sequence
 * @param len The number of colors to compose
 * @return The color resulted from the operation
 */
CODE_TYPE compose(const CODE_TYPE* color_sequence, const int len);

/**
 * Compose a compressed sequence of colors
 * @param sequence The address of the sequence
 * @param len The number of colors to compose
 * @param offset The offset from the compressed sequence's address (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The color resulted from the operation
 */
CODE_TYPE compose_compressed(const CODE_TYPE* color_sequence, const int len, const int offset);

/**
 * Finds the base at position len from a DNA sequence, knowing the first base in the sequence and the sequence of colors up to the wanted base
 * @param start The start base code
 * @param color_sequence The color sequence between the two bases
 * @param len The position of the wanted base code, relative to start which is considered at position 0
 * @return The code of the base at position pos
 */

CODE_TYPE base_color_transformation(const CODE_TYPE start, const CODE_TYPE* color_sequence, const int pos);

/**
 * Finds the base at position len from a DNA sequence, knowing the first base in the sequence and the compressed sequence of colors up to the wanted base
 * @param start The start base code
 * @param color_sequence The color sequence between the two bases
 * @param len The position of the wanted base code, relative to start which is considered at position 0
 * @param offset The offset from the compressed sequence's address (as number of code locations to skip; e.g: offset = 2 for skipping two codes, that take up 4 bits)
 * @return The code of the base at position pos
 */
CODE_TYPE base_compressed_color_transformation(const CODE_TYPE start, const CODE_TYPE* color_sequence, const int pos, const int offset);


#endif

/**
 * Displays an (uncompressed) code sequence
 * @param sequence
 * @param len The number of codes in the sequence (not the length of the vector!)
 */
void display_code_sequence(const CODE_TYPE* sequence, const int len, FILE* f);

/**
 * Displays a compressed code sequence
 * @param sequence
 * @param len The number of codes in the sequence (not the length of the vector!)
 */
void display_compressed_code_sequence(const CODE_TYPE* sequence, const int len, FILE* f);

/**
 * Obtain the reverse complementary of a sequence (which, in the color code, is simply the reversed color sequence)
 */
void sequence__reverse(const CODE_TYPE* sequence, CODE_TYPE* dest, const int len);
/**
 * Obtain the reverse complementary of a sequence (which, in the color code, is simply the reversed color sequence)
 */
void sequence__reverse_compressed(const CODE_TYPE* sequence, CODE_TYPE* dest, const int len);

#endif /* _CODES_H_ */
