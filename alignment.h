#ifndef _ALIGNMENT_H_
#define _ALIGNMENT_H_

#include "codes.h"
#include "read_quality.h"

/* -------------------------------------------------------
 * Score
 */

typedef int ScoreType;


/* -------------------------------------------------------
 * Alignment cell 'metadata': direction of the current partial alignment
 */
typedef char DirectionType;

/* -------------------------------------------------------
 * Alignment parameters
 */

/* Default alignment parameters (scores, penalties) */
#define MATCH (+5)
#define MISMATCH (-4)
#define GAP_OPEN (-9)
#define GAP_EXTEND (-4)
#define ALLOWED_INDELS 3

/**
 * The number of allowed indels relates to the SIMD filter and the possible size of the bandwidth alignment.
 */
#ifdef __AVX2__
#define INDEL_DATA_VECTOR_SIZE 16
#else
#ifdef __SSE2__
#define INDEL_DATA_VECTOR_SIZE 16
#else
#ifdef __SSE__
#define INDEL_DATA_VECTOR_SIZE 8
#else
/* No filter. Default to 7 indels (above this, the alignment algorithm would be much too slow). */
#define INDEL_DATA_VECTOR_SIZE 8
#endif
#endif
#endif

#define INDEL_COUNT_LIMIT (INDEL_DATA_VECTOR_SIZE - 1)

#define SCALE_SCORE(score, quality_level) (ScoreType)(   ((score) * (quality_level+1) + 0.) / READ_QUALITY_LEVELS)

/* For accessing various parameters */
#define GAP_TYPES_SIZE 2
#define GAP_OPEN_IDX   1
#define GAP_EXTEND_IDX 0

#define SYMBOL_PAIR_TYPES_SIZE   2
#define SYMBOL_PAIR_MATCH_IDX    0
#define SYMBOL_PAIR_MISMATCH_IDX 1

/* Symbols used for displaying the alignment */
#define SEQ_GAP_SYMBOL '-'

/* Used for pre-computed alignment parameter quick access: */
#define INDEX_FOR(code) ((code)?1:0)

/* Used as default "bad" score, or as the worst penalty (must be a strong negative) */
#define MIN_SCORE (-16383)

/* Score threshold */
#define MIN_ACCEPTED_SCORE 115
#define MIN_ACCEPTED_SCORE_SIMD_FILTER 100

extern ScoreType min_accepted_score;
extern ScoreType min_accepted_score_simd_filter;

/* For traceback: 4bit codes for various types of pairs : two 4bits codes are stored in one byte */
/* mask for separating upped/lower part of each pair */
#define TRACEBACK_PAIR_MASK           0xC /* 11XX */
#define TRACEBACK_CODE_MASK           0x3 /* XX11 */
/* codes for traceback : note that some can overlap letter codes !!! */
#define TRACEBACK_PAIR_MATCH          0x3 /* 0         */
#define TRACEBACK_PAIR_CODE_MISMATCH  0x4 /* 01XX (XX = color code in the read) */
#ifndef NUCLEOTIDES
#define TRACEBACK_PAIR_COLORBASE_MISMATCH  0x8 /* 10XX (XX = color code in the read) */
#endif
#define TRACEBACK_PAIR_READ_INSERTION      0xC /* 11XX (XX = color code inserted in the read)*/
#define TRACEBACK_PAIR_READ_DELETION       0x1 /* 0001 */
#define TRACEBACK_PAIR_UNDEFINED           0x0 /* 0000 */

/* wrap colors in their backtracing codes */
#define TRACEBACK_MATCH                     (TRACEBACK_PAIR_MATCH)
#ifndef NUCLEOTIDES
#define TRACEBACK_COLORBASE_MISMATCH(code)  (TRACEBACK_PAIR_COLORBASE_MISMATCH | (code))
#endif
#define TRACEBACK_CODE_MISMATCH(code)       (TRACEBACK_PAIR_CODE_MISMATCH      | (code))
#define TRACEBACK_INSERT(code)              (TRACEBACK_PAIR_READ_INSERTION     | (code))
#define TRACEBACK_DELETE                    (TRACEBACK_PAIR_READ_DELETION)

/* access : two 4bits codes are stored in one byte */
#define TRACEBACK_SEQ_SIZE(size) (((size) + 1) >> 1)
#define TRACEBACK_SEQ_GET(seq, i) ((seq[(i) >> 1] >> (((i) & (1)) ? 0 : 4)) & 0xF)
#define TRACEBACK_SEQ_SET(seq, i, code) seq[(i) >> 1] = ((i) & 1) ? ((seq[(i) >> 1] & 0xF0) | ((code) & 0x0F)) : ((seq[(i) >> 1] & 0x0F) | (((code) << 4) & 0xF0))
#define TRACEBACK_SEQ_FILL(seq, len, code, iterator) for (iterator = 0; iterator < len; ++iterator) {TRACEBACK_SEQ_SET(seq, i, code);}


#ifdef NUCLEOTIDES
#define GET_PAIR_TYPE(code) ((((code) & TRACEBACK_PAIR_MASK) == TRACEBACK_PAIR_CODE_MISMATCH)       ? TRACEBACK_PAIR_CODE_MISMATCH       : \
                             (((code) & TRACEBACK_PAIR_MASK) == TRACEBACK_PAIR_READ_INSERTION)      ? TRACEBACK_PAIR_READ_INSERTION      : (code))
#else
#define GET_PAIR_TYPE(code) ((((code) & TRACEBACK_PAIR_MASK) == TRACEBACK_PAIR_CODE_MISMATCH)       ? TRACEBACK_PAIR_CODE_MISMATCH       : \
                            ((((code) & TRACEBACK_PAIR_MASK) == TRACEBACK_PAIR_COLORBASE_MISMATCH)  ? TRACEBACK_PAIR_COLORBASE_MISMATCH  : \
                            ((((code) & TRACEBACK_PAIR_MASK) == TRACEBACK_PAIR_READ_INSERTION)      ? TRACEBACK_PAIR_READ_INSERTION      : (code))))
#endif


#ifdef NUCLEOTIDES
#define GET_PAIR_TYPE_SYMBOL(code) ((((code)                        == TRACEBACK_PAIR_MATCH)               ? 'M' : \
                                   ((((code) & TRACEBACK_PAIR_MASK) == TRACEBACK_PAIR_CODE_MISMATCH)       ? 'S' : \
                                   ((((code) & TRACEBACK_PAIR_MASK) == TRACEBACK_PAIR_READ_INSERTION)      ? 'I' : \
                                   (( (code)                        == TRACEBACK_PAIR_READ_DELETION)       ? 'D' : ('?' + code))))))
#else
#define GET_PAIR_TYPE_SYMBOL(code) (( (code)                        == TRACEBACK_PAIR_MATCH)               ? 'M' : \
                                   ((((code) & TRACEBACK_PAIR_MASK) == TRACEBACK_PAIR_CODE_MISMATCH)       ? 'E' : \
                                   ((((code) & TRACEBACK_PAIR_MASK) == TRACEBACK_PAIR_COLORBASE_MISMATCH)  ? 'S' : \
                                   ((((code) & TRACEBACK_PAIR_MASK) == TRACEBACK_PAIR_READ_INSERTION)      ? 'I' : \
                                   (( (code)                        == TRACEBACK_PAIR_READ_DELETION)       ? 'D' : ('?' + code))))))
#endif


#define TRACEBACK_DISPLAY(sequence, length, output) {                                                                                 \
    fprintf(output, " ");                                                                                                             \
    int _i;                                                                                                                           \
    for (_i = 0; _i < (length); ++_i) {                                                                                               \
      fprintf(output, "%c",                                                                                                           \
              GET_PAIR_TYPE_SYMBOL(TRACEBACK_SEQ_GET((sequence), _i)));                                                               \
    }                                                                                                                                 \
    fprintf(output, "\n ");                                                                                                           \
    for (_i = 0; _i < length; ++_i) {                                                                                                 \
      unsigned char __pts = GET_PAIR_TYPE_SYMBOL(TRACEBACK_SEQ_GET(sequence, _i));                                                    \
      fprintf(output, "%c", (__pts == 'M' || __pts == 'D')  ? ' ' : ('0' + (TRACEBACK_CODE_MASK & TRACEBACK_SEQ_GET(sequence, _i)))); \
    }                                                                                                                                 \
}


/**
 * @struct AlignmentParamType
 * keep the scoring parameters
 */
typedef struct AlignmentParamType {
  /** Opening, extension... */
  ScoreType gap[GAP_TYPES_SIZE];
  /** Match and mismatch scores are pre-computed for several quality intervals
   * by the __set_params function.
   */
  ScoreType pair_scores[SYMBOL_PAIR_TYPES_SIZE][READ_QUALITY_LEVELS];
  /** Number of allowed indels */
  short allowed_indels;
} AlignmentParamType;


#ifndef NUCLEOTIDES
/* Value given to the color_mismatch_freshness field in the alignment cells when
 * there is no uncorrected color mismatch on the path, or the last color mismatch is too far :
 */
#define COLOR_MISMATCH_NONE          0
/* Value given to the color_mismatch_freshness field in the alignment cells where
 * there was a color mismatch that changed the checksum from CODE_IDENTITY.
 * This value 'wears off' in each as the path continues without any correction.
 */
#define COLOR_MISMATCH_NOW           4
#endif

/**
 * @struct AlignmentCellType
 * The data held in a cell from the alignment matrix
 */
typedef struct AlignmentCellType {
  /** Partial score of the partial alignment that ends in this cell */
  ScoreType score;
#ifndef NUCLEOTIDES
  /** To differentiate (for example) two reading errors from a SNP*/
  CODE_TYPE color_checksum;
#endif
  /** For backtracking: relative coordinates of the cell that contributed to this score */
  DirectionType from;
  /** Number of indels existing so far on the path leading to this cell*/
  short indels;
#ifndef NUCLEOTIDES
  /** If at some point there was a color mismatch, this field states how fresh it is; if the mismatch was
   * many cases ago in the path, we consider it is unlikely to be part of an SNP sequence and treat it
   * as a misread color. Therefore, we no longer expect a color correction pair on the path, and instead
   * correct the color checksum automatically */
  short color_mismatch_freshness;
  /** The code used for automatic correction in the case described above.
   * Note: the neutral (and default) code is BLUE (=CODE_IDENTITY), and each color is its own inverse. */
  CODE_TYPE color_autocorrect;
#endif
} AlignmentCellType;

/* ------------------------------------------------------
 * The data necessary to build an alignment
 */

/**
 * @struct
 * The alignment data structure
 */
typedef struct AlignmentType {
  /** Parameters: match, mismatch, gap */
  AlignmentParamType params;
  /** Sizes of the aligned sequences */
  short read_len, ref_len;
  /** The read */
  CODE_TYPE* read;
  /** Its quality */
  ScoreType* read_quality;
  /** The reference sequence */
  CODE_TYPE* reference;
  /** The reference masked (N) */
  CODE_TYPE* reference_masked;
  /** The alignment matrix ("allocated" one) */
  AlignmentCellType** matrix_alloc;
  /** The alignment matrix */
  AlignmentCellType** matrix;
  /** The matrix is not entirely stored into memory; the range fields tells which parts are stored from each line */
  short* range_first;
  short  range_shift;
  /** Coordinates of the cell where the best scoring alignment ends, set by the __align function */
  short best_i, best_j;
  /** Coordinates of the cell where the best scoring alignment starts, set by the __traceback function  */
  short best_i0, best_j0;
  /** Representation of the alignment, set (and returned) by the __traceback function */
  char* to_display;
  /** traceback sequence of bytes filled with the TRACEBACK_MATCH(...), TRACEBACK_CODE_MISMATCH(...),
   *  TRACEBACK_BASE_MISMATCH(...), TRACEBACK_DELETE(...) and TRACEBACK_INSERT(...) macros
   */
  unsigned char* traceback;
  /** traceback length */
  int traceback_seq_len;
} AlignmentType;

/**
 * Alignment structures creation and initialization
 * @param read Read sequence
 * @param read_qualiry Read sequence quality
 * @param read_len Read length
 * @param reference Reference sequence, to which the read is aligned
 * @param read_len Reference sequence length
 * @param allowed_indels Max number of allowed indels in the alignment; usually small, it prevents from computing the entire DP matrix, thus the reducing computation time.
 * @return The address of the alignment data structure
 */
AlignmentType* alignment__create(short read_len, short ref_len, short allowed_indels);

/**
 * Alignment structures creation and initialization, for just one use (read and reference are already provided)
 * @param read Read sequence
 * @param read_quality Read sequence quality
 * @param read_len Read length
 * @param reference Reference sequence, to which the read is aligned
 * @param reference_masked Reference sequence mask, to which the read is aligned
 * @param allowed_indels Max number of allowed indels in the alignment; usually small, it prevents from computing the entire DP matrix, thus the reducing computation time.
 * @return The address of the alignment data structure
 */
AlignmentType* alignment__create_single(CODE_TYPE* read, short* read_quality, short read_len, CODE_TYPE* reference, CODE_TYPE* reference_masked, short ref_len, short allowed_indels);

/**
 * Alignment structures creation and initialization
 * @param alignment
 * @param read_len Read length
 * @param read_len Reference sequence length
 * @param allowed_indels Max number of allowed indels in the alignment; usually small, it prevents from computing the entire DP matrix, thus the reducing computation time.
 */
void alignment__init(AlignmentType* alignment, short read_len, short ref_len, short allowed_indels);

/**
 * Alignment algorithm parameter settings.
 * @param alignment The address of the alignment data structure
 * @param match The match score
 * @param mismatch The mismatch penalty
 * @param gap_open The penalty for gap opening
 * @param gap_extend The penalty for gap extension
 */
void alignment__set_params(AlignmentType * alignment, short match, short mismatch, short gap_open, short gap_extend);

/**
 * Perform the actual alignment
 * @param alignment The address of the alignment data structure
 * @param min_accepted Give up the alignment as soon as it can be shown that it has no chane to reach this minimum imposed score
 * @return The score of the best alignment
 */
int alignment__align(AlignmentType * alignment, ScoreType min_accepted);

/**
 * Reconstructs the alignment from the matrix
 * @param alignment The address of the alignment data structure
 * @param i The line number of the best scoring cell (where the best alignment ends)
 * @param j The column number of the best scoring cell (where the best alignment ends)
 * @return A string containing the representation of this alignment (which the to_display field in the alignment structure)
 */
char* alignment__traceback(AlignmentType *alignment);

/**
 * Reuse the alignment data structure, when the read and reference have the same lengths as the previous ones aligned.
 * @param alignment The address of the data structure to reuse
 * @param read The new read sequence, known to have the length alignment->read_len
 * @param read_quality The new read quality sequence, known to have the length alignment->read_len
 * @param reference The new reference sequence, known to have the length alignment->ref_len
 * @param reference_masked The new reference sequence mask, known to have the length alignment->ref_len
 */
void alignment__reset(AlignmentType *alignment, CODE_TYPE* read, short* read_quality, CODE_TYPE* reference, CODE_TYPE* reference_masked);

/**
 * Reuse the alignment data structure, when the read and reference have the same lengths as the previous ones aligned, and come from compressed sources.
 * @param alignment The address of the data structure to reuse
 * @param read The new read sequence, known to have the length alignment->read_len
 * @param read_quality The new read quality sequence, known to have the length alignment->read_len
 * @param reference The new reference sequence, known to have the length alignment->ref_len
 * @param reference_masked The new reference sequence mask, known to have the length alignment->ref_len
 * @param offset How many codes to skip in the compressed reference and reference_masked
 */
void alignment__reset_with_compressed(AlignmentType *alignment, CODE_TYPE* read, QUAL_TYPE* read_quality, CODE_TYPE* reference, CODE_TYPE* reference_masked, int ref_offset);

/**
 * Reuse the alignment data structure, for a new read that has the same length as the previous one, and comes from a compressed source.
 * the matrix is not initialized, as the new reference is also ewpected.
 * @param alignment The address of the data structure to reuse
 * @param reference The new reference sequence, known to have the length alignment->ref_len
 * @param offset How many codes to skip in the compressed reference
 */
void alignment__reset_with_compressed_read(AlignmentType *alignment,CODE_TYPE* read, QUAL_TYPE* read_quality) ;

/**
 * Reuse the alignment data structure, when the read stays the same and reference has the same lengths as the previous one, and comes from a compressed source.
 * @param alignment The address of the data structure to reuse
 * @param reference The new reference sequence, known to have the length alignment->ref_len
 * @param reference_masked The new reference mask sequence, known to have the length alignment->ref_len
 * @param offset How many codes to skip in the compressed reference
 */
void alignment__reset_with_compressed_ref(AlignmentType *alignment, CODE_TYPE* reference, CODE_TYPE* reference_masked, int ref_offset);

/**
 * Access the alignment matrix, with coordinates that are not "aware" of the matrix' sparsness
 * @param alignment The address of the alignment data structure
 * @param i The requested cell's line coordinate
 * @param j The requested cell's column coordinate, as a number from 0 to reference length (to be transformed in actual matrix coords)
 * @return A reference to the requested matrix cell, or NULL if the parameters are out of range
 */
AlignmentCellType* alignment__matrix_access(AlignmentType * alignment, int i, int j);

/**
 * Retrieve a score in the alignment matrix, with coordinates that are not "aware" of the matrix' sparseness
 * @param alignment The address of the alignment data structure
 * @param i The requested cell's line coordinate
 * @param j The requested cell's column coordinate, as a number from 0 to reference length (to be transformed in actual matrix coords)
 * @return The found score, or MIN_SCORE if the parameters are out of range
 */
ScoreType alignment__get_score(const AlignmentType * alignment, int i, int j);

/**
 * Frees the memory of an alignment
 * @param alignment The address of the data structure to be disposed of
 */
void alignment__destroy(AlignmentType *alignment);

/**
 * Generates the CIGAR string and the right DNA sequence (no reading errors) from
 * the code sequence alignment.
 *
 * @param traceback a compressed representation of the alignment between the read and the reference
 * @param traceback_len the length of this alignment
 * @param ref_sequence pointer to the beginning of the reference fragment aligned to the read
 * @param offset the offset of the beginning
 * @param ref_start_symbol the first symbol of the reference sequence
 * @param CIGAR_dest the CIGAR representation will be written here
 * @param SEQ_dest the DNA sequence will be written here
 * @return The edit distance
 */
int  alignment__traceback_to_CIGAR(const unsigned char* traceback, const int traceback_len,
                                   const CODE_TYPE* ref_sequence, const int offset,
#ifndef NUCLEOTIDES
                                   const char ref_start_symbol,
#endif
                                   char* CIGAR_dest, char* SEQ_dest);

#endif /* _ALIGNMENT_H_ */
