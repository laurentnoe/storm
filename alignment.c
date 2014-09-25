#include "alignment.h"
#include "run.h"

#include <math.h>
/* Symbols used for displaying the alignment */
#define DISPLAYED_MATCH_SYMBOL '|'
#define DISPLAYED_MISMATCH_SYMBOL '.'
#define DISPLAYED_SEQ_GAP_SYMBOL '-'
#define DISPLAYED_MATCH_GAP_SYMBOL ' '


/* Encoding/Decoding of the 'from' field */
#define I_MASK 0x2
#define J_MASK 0x1
#define FROM_LEFT(direction)   (!((direction) & I_MASK) &&  ((direction) & J_MASK))
#define FROM_TOP(direction)    ( ((direction) & I_MASK) && !((direction) & J_MASK))
#define FROM_CORNER(direction) ( ((direction) & I_MASK) &&  ((direction) & J_MASK))
#define D_I(direction) (-(((direction) & I_MASK) >> 1))
#define D_J(direction) (-( (direction) & J_MASK))

#ifndef NUCLETIDES
#define COLOR_MISMATCH_PERSISTENCE(q_level) ((q_level)) /*(((q_level) + 1) >> 1)*/
#endif

/* 'Shortcuts' for some frequently used code blocks */
#define CHECK_RELATIVE_COORDS(i, j) ((i) >= 0 && (i) < alignment->read_len && (j) >= alignment->range_first[i] && (j) <= alignment->range_first[i] + alignment->range_shift)

#define CHECK_ABSOLUTE_COORDS(i, j) ((i) >= 0 && (i) < alignment->read_len && (j) >= 0 && (j) <= alignment->range_shift)

#define REF_START(i)    (alignment->range_first[i])
#define REF_END(i)      (alignment->range_first[i] + alignment->range_shift)
#define REF_LINE_LEN    (alignment->range_shift+1)
#define TO_MATRX_ABS_COORD(j, i) ((j) - alignment->range_first[i])
#define TO_REFERENCE_COORD(j, i) ((j) + alignment->range_first[i])

ScoreType min_accepted_score = MIN_ACCEPTED_SCORE;
ScoreType min_accepted_score_simd_filter = MIN_ACCEPTED_SCORE_SIMD_FILTER;

/**
 * Access the alignment matrix, with coordinates that are not "aware" of the matrix' sparsness
 * @param alignment The address of the alignment data structure
 * @param i The requested cell's line coordinate
 * @param j The requested cell's column coordinate, as a number from 0 to reference length (to be transformed in actual matrix coords)
 * @return A reference to the requested matrix cell, or NULL if the parameters are out of range
 */
inline AlignmentCellType* alignment__matrix_access(AlignmentType * alignment, int i, int j) {
  return (CHECK_RELATIVE_COORDS(i, j)) ? (alignment->matrix[i] + TO_MATRX_ABS_COORD(j, i)) : NULL;
}
/**
 * Access the alignment matrix, with coordinates that are not "aware" of the matrix' sparsness, without checking the parameter range
 * @param alignment The address of the alignment data structure
 * @param i The requested cell's line coordinate
 * @param j The requested cell's column coordinate, as a number from 0 to reference length (to be transformed in actual matrix coords)
 * @return A reference to the requested matrix cell,
 */
inline AlignmentCellType* alignment__matrix_access__no_check(AlignmentType * alignment, int i, int j) {
  return alignment->matrix[i] + TO_MATRX_ABS_COORD(j, i);
}

/**
 * Access the alignment matrix, with the true matrix coordinates
 * @param alignment The address of the alignment data structure
 * @param i The requested cell's line coordinate
 * @param j The requested cell's column coordinate, as a number from 0 to matrix line length
 * @return A reference to the requested matrix cell, or NULL if the parameters are out of range
 */
inline AlignmentCellType* alignment__matrix_absolute_access(AlignmentType * alignment, int i, int j) {
  return (CHECK_ABSOLUTE_COORDS(i, j)) ? (alignment->matrix[i] + j) : NULL;
}
/**
 * Access the alignment matrix, with the true matrix coordinates, without checking the parameter range
 * @param alignment The address of the alignment data structure
 * @param i The requested cell's line coordinate
 * @param j The requested cell's column coordinate, as a number from 0 to matrix line length
 * @return A reference to the requested matrix cell, or NULL if the parameters are out of range
 */
inline AlignmentCellType* alignment__matrix_absolute_access__no_check(AlignmentType * alignment, int i, int j) {
  return alignment->matrix[i] + j;
}
/**
 * Retrieve a score in the alignment matrix, with coordinates that are not "aware" of the matrix' sparseness
 * @param alignment The address of the alignment data structure
 * @param i The requested cell's line coordinate
 * @param j The requested cell's column coordinate, as a number from 0 to reference length (to be transformed in actual matrix coords)
 * @return The found score, or MIN_SCORE if the parameters are out of range
 */
inline ScoreType alignment__get_score(const AlignmentType * alignment, int i, int j) {
  return (CHECK_RELATIVE_COORDS(i, j)) ? alignment->matrix[i][TO_MATRX_ABS_COORD(j, i)].score : MIN_SCORE;
}
/**
 * Retrieve a score in the alignment matrix, with coordinates that are not "aware" of the matrix' sparseness, without checking the parameter range
 * @param alignment The address of the alignment data structure
 * @param i The requested cell's line coordinate
 * @param j The requested cell's column coordinate, as a number from 0 to reference length (to be transformed in actual matrix coords)
 * @return The found score, or MIN_SCORE if the parameters are out of range
 */
inline ScoreType alignment__get_score__no_check(const AlignmentType * alignment, int i, int j) {
  return alignment->matrix[i][TO_MATRX_ABS_COORD(j, i)].score;
}

/**
 * Retrieve a score in the alignment matrix, with with the true matrix coordinates
 * @param alignment The address of the alignment data structure
 * @param i The requested cell's line coordinate
 * @param j The requested cell's column coordinate, as a number from 0 to matrix line length
 * @return The found score, or MIN_SCORE if the parameters are out of range
 */
inline ScoreType alignment__get_score_absolute(const AlignmentType * alignment, int i, int j) {
  return (CHECK_ABSOLUTE_COORDS(i, j)) ? alignment->matrix[i][j].score : MIN_SCORE;
}
/**
 * Retrieve a score in the alignment matrix, with with the true matrix coordinates, without checking the parameter range
 * @param alignment The address of the alignment data structure
 * @param i The requested cell's line coordinate
 * @param j The requested cell's column coordinate, as a number from 0 to matrix line length
 * @return The found score, or MIN_SCORE if the parameters are out of range
 */
inline ScoreType alignment__get_score_absolute__no_check(const AlignmentType * alignment, int i, int j) {
  return alignment->matrix[i][j].score;
}

/**
 * Initialize the read_quality vector
 * The memory is considered allocated, and all the sizes correct (i.e. read_len)
 * @param alignment
 * @param read_quality The read quality 'classes' (levels), short integers ranging from 0 to READ_QUALITY_LEVELS
 */
inline void alignment__init_read_quality(AlignmentType * alignment, short* read_quality) {
  int i;
  if (read_quality != NULL) {
    for (i = 0; i <  alignment->read_len; ++i) {
      alignment->read_quality[i] = QUALITY_LEVEL(MIN(MAX_READ_QUALITY, read_quality[i]));
    }
  } else {
    for (i = 0; i <  alignment->read_len; ++i) {
      alignment->read_quality[i] = QUALITY_LEVEL(MAX_READ_QUALITY);
    }
  }
}

/**
 * Initialize the read_quality vector, from a compressed source
 * The memory is considered allocated, and all the sizes correct (i.e. read_len)
 * @param alignment
 * @param read_quality The read quality 'classes' (levels), short integers ranging from 0 to READ_QUALITY_LEVELS
 */
inline void alignment__init_read_quality_compressed(AlignmentType * alignment, QUAL_TYPE* read_quality) {
  int i;
  if (read_quality != NULL) {
    for (i = 0; i <  alignment->read_len; ++i) {
      alignment->read_quality[i] = NTH_QUAL(read_quality, i);
    }
  } else {
    for (i = 0; i <  alignment->read_len; ++i) {
      alignment->read_quality[i] = QUALITY_LEVEL(MAX_READ_QUALITY);
    }
  }
}

/**
 * Reuse the alignment data structure, when the read and reference have the same lengths as the previous ones aligned.
 * @param alignment The address of the data structure to reuse
 * @param read The new read sequence, known to have the length alignment->read_len
 * @param read_quality The new read quality sequence, known to have the length alignment->read_len
 * @param reference The new reference sequence, known to have the length alignment->ref_len
 * @param reference_masked The new reference sequence mask, known to have the length alignment->ref_len
 */
inline void alignment__reset(AlignmentType *alignment, CODE_TYPE* read, short* read_quality, CODE_TYPE* reference, CODE_TYPE* reference_masked){
  memcpy(alignment->read,             read,             alignment->read_len * sizeof(CODE_TYPE));
  memcpy(alignment->reference,        reference,        alignment->ref_len  * sizeof(CODE_TYPE));
  memcpy(alignment->reference_masked, reference_masked, alignment->ref_len  * sizeof(CODE_TYPE));
  alignment__init_read_quality(alignment, read_quality);
}
/**
 * Reuse the alignment data structure, when the read and reference have the same lengths as the previous ones aligned, and come from compressed sources.
 * @param alignment The address of the data structure to reuse
 * @param read The new read sequence, known to have the length alignment->read_len
 * @param read_quality The new read quality sequence, known to have the length alignment->read_len
 * @param reference The new reference sequence, known to have the length alignment->ref_len
 * @param reference_masked The new reference sequence mask, known to have the length alignment->ref_len
 * @param offset How many codes to skip in the compressed reference and reference_masked
 */
inline void alignment__reset_with_compressed(AlignmentType *alignment, CODE_TYPE* read, QUAL_TYPE* read_quality, CODE_TYPE* reference, CODE_TYPE* reference_masked, int ref_offset){
  decompress_to(read,             alignment->read,             alignment->read_len, 0);
  decompress_to(reference,        alignment->reference,        alignment->ref_len,  ref_offset);
  decompress_to(reference_masked, alignment->reference_masked, alignment->ref_len,  ref_offset);
  alignment__init_read_quality_compressed(alignment, read_quality);
}

/**
 * Deduces the score of an aligned pair of colors, composed as color_match
 * @param alignment The address of the alignment data structure
 * @param color_match The composition of the two colors
 * @param quality_level The quality level of the read
 * @return The corresponding score
 */
inline ScoreType alignment__score_for_match(AlignmentType * alignment, CODE_TYPE match, short quality_level) {
  return alignment->params.pair_scores[INDEX_FOR(match)][quality_level] ;
}

/**
 * Actual construction of the alignment "matrix", more precisely a part of it.
 * The read sequence symbols label the matrix lines, while the reference sequence symbols index the columns. On each line, only some columns are actually allocated: because of the small number of allowed indels, each symbol in the read has a limited number of allowed pairs in the reference; their range is memorized in the 'range' vector.
 * @param alignment The address of the alignment data structures
 * @param allowed_indels The maximum number of indels allowed in the alignment.
 */
static inline void alignment__create_matrix(AlignmentType * alignment, int allowed_indels) {
  /* Few indels allowed: no need to compute the entire DP matrix, just a neighborhood of the diagonal */

  /* Create the alignment matrix */
  /* consider the reference on top and the read on the bottom (side))*/
  SAFE_FAILURE__ALLOC(alignment->matrix_alloc, alignment->read_len, AlignmentCellType*);
  SAFE_FAILURE__ALLOC(alignment->matrix,       alignment->read_len, AlignmentCellType*);
  SAFE_FAILURE__ALLOC(alignment->range_first,  alignment->read_len, short);
  int mat_k;
  for (mat_k = 0; mat_k < alignment->read_len; ++mat_k) {
    /* where the overlap starts in the reference sequence
     */
    alignment->range_first[mat_k] = mat_k;
    /* line length
     */
    alignment->range_shift = 2*allowed_indels;

    /* each line has 2 extra cells allocated, one on the left and one on the right, each holding the minimum possible score,
     * with the sole purpose of allowing the same recurrence formula for all cells, including those on the borders
     */

    SAFE_FAILURE__ALLOC(alignment->matrix_alloc[mat_k], REF_LINE_LEN + 2, AlignmentCellType);
    alignment->matrix[mat_k] =  alignment->matrix_alloc[mat_k] + 1;
    /* 'fake' left border cell, that never changes */
    (alignment->matrix_alloc[mat_k][0]).score      = MIN_SCORE;
    (alignment->matrix_alloc[mat_k][0]).indels     = allowed_indels;
    (alignment->matrix_alloc[mat_k][0]).from       = I_MASK | J_MASK;
    /* 'fake' right border cell, that never changes */
    (alignment->matrix_alloc[mat_k][REF_LINE_LEN + 1]).score      = MIN_SCORE;
    (alignment->matrix_alloc[mat_k][REF_LINE_LEN + 1]).indels     = allowed_indels;
    (alignment->matrix_alloc[mat_k][REF_LINE_LEN + 1]).from       = I_MASK | J_MASK;
    /* the direction masks and indel count are null */
  }
  /* alignment__init_matrix(alignment); */
}

/**
 * Alignment algorithm parameter settings.
 * @param alignment The address of the alignment data structure
 * @param match The match score
 * @param mismatch The mismatch penalty
 * @param gap_open The penalty for gap opening
 * @param gap_extend The penalty for gap extension
 */
void alignment__set_params(AlignmentType * alignment, short match, short mismatch, short gap_open, short gap_extend) {
  int i;
  alignment->params.gap[GAP_OPEN_IDX]   = gap_open;
  alignment->params.gap[GAP_EXTEND_IDX] = gap_extend;
  /*
  * Pre-compute match/mismatch scores for several quality intervals.
  * Phred quality, log transformed error probability: Q = -10 log10(P), where P is the error probability
  * (http://www.phrap.org/phredphrapconsed.html)
  * => P = 10^-(Q/10)
  *
  * In the pair_scores table, the second index (i) has the following relation to the quality value:
  * (READ_QUALITY_LEVELS - 1) * min (Q, 40) / 40 => index of the score
  * (beyond Q=40, the error probability is less than 0.0001, and is considered insignificant)
  *
  * For match, the lower is the quality, the lower is the reward:
  * i / (float)READ_QUALITY_LEVELS * matchscore
  * Similarly, For mismatch, the lower is the quality, the lower is the mismatch penalty:
  * i / (float)READ_QUALITY_LEVELS * mismatchpenalty
  *
  */
  for (i = 0; i < READ_QUALITY_LEVELS; ++i) {
    alignment->params.pair_scores[SYMBOL_PAIR_MATCH_IDX][i]    = SCALE_SCORE(   match, i);
    alignment->params.pair_scores[SYMBOL_PAIR_MISMATCH_IDX][i] = SCALE_SCORE(mismatch, i);
  }
  /* should be followed by __init_matrix for proper effect */
  /*
   *if (alignment->matrix) {
   *  alignment__init_matrix(alignment);
   *}
   */
}

/**
 * Alignment structures creation and initialization
 * @param alignment
 * @param read_len Read length
 * @param read_len Reference sequence length
 * @param allowed_indels Max number of allowed indels in the alignment; usually small, it prevents from computing the entire DP matrix, thus the reducing computation time.
 */
inline void alignment__init(AlignmentType* alignment, short read_len, short ref_len, short allowed_indels) {
  /* what is constructed and "returned" */
  alignment->params.allowed_indels = allowed_indels;

  alignment__set_params(alignment, DEFAULT_MATCH, DEFAULT_MISMATCH, DEFAULT_GAP_OPEN, DEFAULT_GAP_EXTEND);

  SAFE_FAILURE__ALLOC(alignment->read, (size_t)read_len, CODE_TYPE);
  alignment->read_len = read_len;

  SAFE_FAILURE__ALLOC(alignment->reference,        (size_t)ref_len, CODE_TYPE);
  SAFE_FAILURE__ALLOC(alignment->reference_masked, (size_t)ref_len, CODE_TYPE);
  alignment->ref_len = ref_len;

  SAFE_FAILURE__ALLOC(alignment->read_quality, (size_t)read_len, ScoreType);

  /* The size of the reference sequence indicates the number of indels accepted; not sure if this is a good idea; TODO to be reviewed */
  /* Reference is considered larger than the read; */
  alignment__create_matrix(alignment, allowed_indels);

  /* Create the string where the alignment is displayed;
     3 times (the worst case alignment length + 1 for '\n') + '\0' */
  SAFE_FAILURE__ALLOC(alignment->to_display, 3*(read_len + ref_len + 1) + 1, char);
  /* Create a traceback (2 symbols / unsigned char) of size
   * (read_len + allowed_indels)
   */
  SAFE_FAILURE__ALLOC(alignment->traceback, TRACEBACK_SEQ_SIZE(alignment->read_len + alignment->params.allowed_indels), unsigned char);
}

/**
 * Initialize the alignment matrix
 * @param alignment The address of the alignment data structures
 */
static inline void alignment__init_matrix(AlignmentType * alignment) {
  /* First line should be this for indels ... */
  int mat_k;
  for (mat_k = 0; mat_k < alignment->read_len; ++mat_k) {
    int j;
    for (j = 0; j <= alignment->range_shift; ++j) {
      if (alignment->reference_masked[TO_REFERENCE_COORD(j, mat_k)] == 0) {
      /* a) if non masked part */
#ifndef NUCLEOTIDES
      (alignment->matrix[mat_k][j]).color_autocorrect = (alignment->matrix[mat_k][j]).color_checksum = COMPOSE(alignment->read[mat_k], alignment->reference[TO_REFERENCE_COORD(j, mat_k)]);
      (alignment->matrix[mat_k][j]).color_mismatch_freshness = ((alignment->matrix[mat_k][j]).color_autocorrect) ? COLOR_MISMATCH_PERSISTENCE(alignment->read_quality[mat_k]) : COLOR_MISMATCH_NONE;
      /*Score initialized with the match / mismatch score adjusted wrt the quality */
      (alignment->matrix[mat_k][j]).score =
        alignment__score_for_match(alignment, alignment->matrix[mat_k][j].color_checksum, alignment->read_quality[mat_k]);
#else
      (alignment->matrix[mat_k][j]).score =
        alignment__score_for_match(alignment, alignment->read[mat_k] != alignment->reference[TO_REFERENCE_COORD(j, mat_k)], alignment->read_quality[mat_k]);
#endif
      } else {
      /* b) masked part */
#ifndef NUCLEOTIDES
      (alignment->matrix[mat_k][j]).color_autocorrect = (alignment->matrix[mat_k][j]).color_checksum = 0;
      (alignment->matrix[mat_k][j]).color_mismatch_freshness = 0;
      (alignment->matrix[mat_k][j]).score = MIN_SCORE;
#else
      (alignment->matrix[mat_k][j]).score = MIN_SCORE;
#endif

      }
      (alignment->matrix[mat_k][j]).from   = I_MASK | J_MASK;
      (alignment->matrix[mat_k][j]).indels = 0;
    }
  }
  alignment->best_i0 = alignment->best_j0 = alignment->best_i = alignment->best_j = -1;
}

/**
 * Alignment structures creation and initialization
 * @param read_len Read length
 * @param read_len Reference sequence length
 * @param allowed_indels Max number of allowed indels in the alignment; usually small, it prevents from computing the entire DP matrix, thus the reducing computation time.
 * @return The address of the alignment data structure
 */
AlignmentType* alignment__create(short read_len, short ref_len, short allowed_indels) {
  /* what is constructed and "returned" */
  AlignmentType * alignment;
  SAFE_FAILURE__ALLOC(alignment, 1, AlignmentType);
  alignment__init(alignment, read_len, ref_len, allowed_indels);
  return alignment;
}

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
AlignmentType* alignment__create_single(CODE_TYPE* read, short* read_quality, short read_len, CODE_TYPE* reference, CODE_TYPE* reference_masked, short ref_len, short allowed_indels) {
  AlignmentType* alignment = alignment__create(read_len, ref_len, allowed_indels);
  alignment__reset(alignment, read, read_quality, reference, reference_masked);
  return alignment;
}

/**
 * Reuse the alignment data structure, for a new read that has the same length as the previous one, and comes from a compressed source.
 * the matrix is not initialized, as the new reference is also ewpected.
 * @param alignment The address of the data structure to reuse
 * @param reference The new reference sequence, known to have the length alignment->ref_len
 * @param offset How many codes to skip in the compressed reference
 */
inline void alignment__reset_with_compressed_read(AlignmentType *alignment, CODE_TYPE* read, QUAL_TYPE* read_quality) {
  decompress_to(read, alignment->read, alignment->read_len, 0);
  alignment__init_read_quality_compressed(alignment, read_quality);
}

/**
 * Reuse the alignment data structure, when the read stays the same and reference has the same lengths as the previous one, and comes from a compressed source.
 * @param alignment The address of the data structure to reuse
 * @param reference The new reference sequence, known to have the length alignment->ref_len
 * @param reference_masked The new reference mask sequence, known to have the length alignment->ref_len
 * @param offset How many codes to skip in the compressed reference
 */
inline void alignment__reset_with_compressed_ref(AlignmentType *alignment, CODE_TYPE* reference, CODE_TYPE* reference_masked, int ref_offset){
  decompress_to(reference,        alignment->reference,        alignment->ref_len,  ref_offset);
  decompress_to(reference_masked, alignment->reference_masked, alignment->ref_len,  ref_offset);
}

/**
 * Frees the memory of an alignment
 * @param alignment The address of the data structure to be disposed of
 */
void alignment__destroy(AlignmentType *alignment) {
  free(alignment->read);
  alignment->read = NULL;
  free(alignment->reference);
  alignment->reference = NULL;
  free(alignment->reference_masked);
  alignment->reference_masked = NULL;
  free(alignment->read_quality);
  alignment->read_quality = NULL;
  int i;
  for (i = 0; i < alignment->read_len; ++i) {
    free(alignment->matrix_alloc[i]);
    alignment->matrix_alloc[i] = NULL;
  }
  free(alignment->matrix);
  alignment->matrix = NULL;
  free(alignment->matrix_alloc);
  alignment->matrix_alloc = NULL;
  free(alignment->range_first);
  alignment->range_first = NULL;
  if (alignment->to_display) {
    free(alignment->to_display);
    alignment->to_display = NULL;
  }
  if (alignment->traceback) {
    free(alignment->traceback);
    alignment->traceback = NULL;
  }
}


#ifndef NUCLEOTIDES

/**
 * In an alignment, if the composition of a series of N color mismatches becomes CODE_IDENTITY, then it corresponds to N-1 base mismatches
 * in the DNA space, and should be penalized less. A color mismatch is not penalized (gets score 0) if it corrects such
 * a series of color mismatches.
 * @param prev Previous code composition
 * @param crt Current match code
 * @param crt_score The score that would normally be given to the symbol pair, regardless of color corrections.
 * @return The score that should be given to the symbol pair wrt the eventual color correction:
 * O (no penalty) if there is a correction, or the previously established score otherwise.
 */

static inline ScoreType alignment__color_correction_score_adjustment(CODE_TYPE prev, CODE_TYPE crt, CODE_TYPE reference, ScoreType crt_score) {
  // TODO : Improve
  return (prev && (COMPOSE(prev, crt) == (CODE_IDENTITY) || COMPOSE(prev, crt) == reference) && crt_score!= MIN_SCORE && crt_score < 0) ? -2*crt_score : crt_score;
}
#endif

/**
 * Suitable for affine gaps. According to the current alignment direction and the previous cell, establishes
 * if the gap is opened or extended.
 * @param alignment
 * @param prev_cell
 * @param direction The relative position of the current cell wrt prev_cell, encoded as I_MASK & vertical_dif | J_MASK & horizontal_dif.
 * @returns According to the case, either the gap_open or the gap_extension penalty, stored in the alignment's parameter field.
 */
inline ScoreType alignment__gap_penalty(AlignmentType * alignment, AlignmentCellType *prev_cell, DirectionType direction) {
  return alignment->params.gap[INDEX_FOR(prev_cell->from ^ direction)];
}

/**
 * Perform the actual gapped alignment.
 * @param alignment The address of the alignment data structure
 * @return The score of the best alignment
 */
inline int alignment__align(AlignmentType * alignment, ScoreType min_accepted) {
  ScoreType best_score, max_on_line;
#ifdef NUCLEOTIDES
  ScoreType best_possible = (alignment->read_len)     * alignment->params.pair_scores[SYMBOL_PAIR_MATCH_IDX][MAX_READ_QUALITY_LEVEL];
#else
  ScoreType best_possible = (alignment->read_len - 2) * alignment->params.pair_scores[SYMBOL_PAIR_MATCH_IDX][MAX_READ_QUALITY_LEVEL];
#endif
  /* 1) Compute the matching / mismatches scores for each cell alone (with help of the quality values) */
  alignment__init_matrix(alignment);

  /* 2) Doing the DP algorithm Forward */
  /* Now starting at 1, since the first line should stay as it was initialized */
  int i;
  for (i = 1; i < alignment->read_len; ++i) {
    max_on_line = MIN_SCORE;
    int j;
    for (j = REF_START(i); j <= REF_END(i); ++j) {
      AlignmentCellType * crt_cell = alignment__matrix_access__no_check(alignment,  i,    j);

      /* Get the 3 cases that can contribute to the current one's score */
      AlignmentCellType * corner   = alignment__matrix_access__no_check(alignment, i-1, j-1);
      AlignmentCellType * top      = alignment__matrix_access__no_check(alignment, i-1, j  );
      AlignmentCellType * left     = alignment__matrix_access__no_check(alignment,   i, j-1);
      AlignmentCellType * it       = corner; /* "Monty Python" reference ?? */

      /* find out which one does contribute */
#ifndef NUCLEOTIDES
      ScoreType match_score = alignment__color_correction_score_adjustment(
          corner->color_checksum, crt_cell->color_checksum, corner->color_autocorrect, crt_cell->score
      );
#else
      ScoreType match_score = crt_cell->score;
#endif
      crt_cell->score = corner->score + match_score;

      ScoreType tmp_score;
      if ((top->indels < alignment->params.allowed_indels) && (tmp_score = top->score + alignment__gap_penalty(alignment, top, I_MASK)) > crt_cell->score) {
        it = top;
        crt_cell->score = tmp_score;
      }
      if ((left->indels < alignment->params.allowed_indels) && (tmp_score = left->score + alignment__gap_penalty(alignment, left, J_MASK)) > crt_cell->score) {
        it = left;
        crt_cell->score = tmp_score;
      }

      if (it == corner) {
        crt_cell->indels = it->indels;
#ifndef NUCLEOTIDES
        crt_cell->color_checksum = COMPOSE(crt_cell->color_checksum, it->color_checksum);
        // TODO optimize
        if (it->color_checksum != crt_cell->color_checksum) { /* mismatch in the crt cell*/
          if (it->color_checksum == CODE_IDENTITY) {
            /* can be the beginning of a SNP or can be a reading error */
            crt_cell->color_autocorrect = crt_cell->color_checksum;
            crt_cell->color_mismatch_freshness = COLOR_MISMATCH_PERSISTENCE(alignment->read_quality[i]);
          } else if (crt_cell->color_checksum == it->color_autocorrect) {
            /* ended a SNP */
            crt_cell->color_autocorrect = CODE_IDENTITY;
            crt_cell->color_checksum = CODE_IDENTITY;
            crt_cell->color_mismatch_freshness = COLOR_MISMATCH_NONE;
          } else if (crt_cell->color_checksum == CODE_IDENTITY) {
            if (it->color_mismatch_freshness - 1 > 0) {
              crt_cell->color_autocorrect = it->color_autocorrect;
              crt_cell->color_mismatch_freshness = it->color_mismatch_freshness - 1;
            } else {
              crt_cell->color_autocorrect = CODE_IDENTITY;
              crt_cell->color_mismatch_freshness = COLOR_MISMATCH_NONE;
            }
          } else { /**/
            if (it->color_mismatch_freshness - 1 > 0) {
              crt_cell->color_autocorrect = it->color_autocorrect;
              crt_cell->color_mismatch_freshness = it->color_mismatch_freshness - 1;
            } else {
              crt_cell->color_autocorrect = crt_cell->color_checksum;
              crt_cell->color_mismatch_freshness = COLOR_MISMATCH_PERSISTENCE(alignment->read_quality[i]);
            }
          }
        } else { /* match */
          if (it->color_autocorrect != CODE_IDENTITY) {
            /* 'deprecate' the previous mismatch */
            if (it->color_mismatch_freshness - 1 > 0) {
              if (crt_cell->color_checksum == CODE_IDENTITY) {
                crt_cell->color_autocorrect = CODE_IDENTITY;
                crt_cell->color_mismatch_freshness = COLOR_MISMATCH_NONE;
              } else {
                crt_cell->color_autocorrect = it->color_autocorrect;
                crt_cell->color_mismatch_freshness = it->color_mismatch_freshness - 1;
              }
            }
          } else {
            crt_cell->color_checksum = CODE_IDENTITY;
            crt_cell->color_autocorrect = CODE_IDENTITY;
            crt_cell->color_mismatch_freshness = COLOR_MISMATCH_NONE;
          }
        }
#endif
      } else if (it == top) {
        crt_cell->from = I_MASK;
        crt_cell->indels = it->indels + 1;
#ifndef NUCLEOTIDES
        crt_cell->color_checksum = COMPOSE(top->color_checksum, alignment->read[i]);
        if (it->color_mismatch_freshness - 1 > 0) {
          crt_cell->color_mismatch_freshness = it->color_mismatch_freshness - 1;
          crt_cell->color_autocorrect = corner->color_autocorrect;
        }
#endif
      } else { /* it == left */
        crt_cell->from = J_MASK;
        crt_cell->indels = it->indels + 1;
#ifndef NUCLEOTIDES
        crt_cell->color_checksum = COMPOSE(left->color_checksum, alignment->reference[j]);
        if (it->color_mismatch_freshness - 1 > 0) {
          crt_cell->color_mismatch_freshness = it->color_mismatch_freshness - 1;
          crt_cell->color_autocorrect = corner->color_autocorrect;
        }
#endif
      }
      if (max_on_line < crt_cell->score) {
        max_on_line = crt_cell->score;
      }
    }
    /* estimate if the alignment has the chance to reach the minimum accepted quality;
     * if so, continue the alignment;
     * otherwise, abort and return MIN_SCORE;
     */
    if (max_on_line + best_possible < min_accepted) {
      //printf("score : %d < %d\n", max_on_line + best_possible, min_accepted);
      return MIN_SCORE;
    }
    best_possible -= alignment->params.pair_scores[SYMBOL_PAIR_MATCH_IDX][MAX_READ_QUALITY_LEVEL];
    //printf("best : %d\n", best_score);
  }
  /* find the maximum on the right bottom border, and retrieve it as the best alignment score */
  best_score = MIN_SCORE;
  i = alignment->read_len - 1;
  int j;
  for (j = REF_START(i); j <= REF_END(i); ++j) {
    if (best_score < alignment__get_score__no_check(alignment, i, j)) {
      best_score = alignment__get_score__no_check(alignment, i, j);
      alignment->best_i = i;
      alignment->best_j = j;
    }
  }
  return best_score;
}

/**
 * Reconstructs the alignment from the matrix
 * @param alignment The address of the alignment data structure
 * @param i The line number of the best scoring cell (where the best alignment ends)
 * @param j The column number of the best scoring cell (where the best alignment ends)
 * @return A string containing the representation of this alignment (which the to_display field in the alignment structure)
 */

char* alignment__traceback(AlignmentType *alignment) {
  AlignmentCellType *cell;
  int i = alignment->best_i0 = alignment->best_i, j = alignment->best_j0 = alignment->best_j;
  size_t al_size = (size_t)(alignment->read_len + alignment->params.allowed_indels + 2); /* worst case (full gapped alignment) + '\n' + '\0' */
  char *read = NULL, *ref = NULL, *m = NULL;

  if (i < 0 || j < 0) {
    return NULL;
  }

#ifndef NUCLEOTIDES
  CODE_TYPE color = CODE_IDENTITY;
#endif

  alignment->traceback_seq_len = 0;

  /* only build the displayed alignment if the verbosity level allows to display it */
  VERB_FILTER(VERBOSITY_HIGH, {
    SAFE_FAILURE__ALLOC(read, al_size, char);
    SAFE_FAILURE__ALLOC(ref, al_size, char);
    SAFE_FAILURE__ALLOC(m, al_size, char);
  });

  int k = 0;
  while ((cell = alignment__matrix_access(alignment, i, j))) {
    alignment->best_i0 = i;
    alignment->best_j0 = j;

    /* only build the displayed alignment if the verbosity level allows to display it */
    VERB_FILTER(VERBOSITY_HIGH, {
      read[k] = (FROM_LEFT(cell->from)) ? DISPLAYED_SEQ_GAP_SYMBOL : CHARDISPLAY(alignment->read[i]);
      ref[k]  = (FROM_TOP(cell->from)) ?  DISPLAYED_SEQ_GAP_SYMBOL : CHARDISPLAY(alignment->reference[j]);
      m[k]    = (FROM_CORNER(cell->from)) ? ((alignment->read[i] == alignment->reference[j]) ? DISPLAYED_MATCH_SYMBOL : DISPLAYED_MISMATCH_SYMBOL) : DISPLAYED_MATCH_GAP_SYMBOL;
      ++k;
    });

    int di = D_I(cell->from);
    int dj = D_J(cell->from);
    /* if there was a gap, mark the position of the indel:
     *  0  => no more indels
     *  N  => insertion in the read at position N-1
     * -N  => deletion  in the read at position N-1 */
    if (di && !dj) {
      TRACEBACK_SEQ_SET(alignment->traceback, alignment->traceback_seq_len, TRACEBACK_INSERT(alignment->read[i]));
    } else if (dj && !di) {
      TRACEBACK_SEQ_SET(alignment->traceback, alignment->traceback_seq_len, TRACEBACK_DELETE);
    } else {
      if (alignment->read[i] != alignment->reference[j]) {
#ifdef NUCLEOTIDES
        TRACEBACK_SEQ_SET(alignment->traceback, alignment->traceback_seq_len, TRACEBACK_CODE_MISMATCH(alignment->read[i]));
#else
        if (color || (i != 0 && j != 0 && alignment__matrix_access(alignment, i, j)->score > alignment__matrix_access(alignment, i-1, j-1)->score)) {
          /* color mismatch */
          TRACEBACK_SEQ_SET(alignment->traceback, alignment->traceback_seq_len, TRACEBACK_COLORBASE_MISMATCH(alignment->read[i]));
          color = COMPOSE(color, COMPOSE(alignment->read[i], alignment->reference[j]));
        } else {
          /* color mismatch corrected (correction of color, after for example a previous color mismatch) : this one costs less than a color mismatch ... */
          TRACEBACK_SEQ_SET(alignment->traceback, alignment->traceback_seq_len, TRACEBACK_CODE_MISMATCH(alignment->read[i]));
        }
#endif
      } else {
        TRACEBACK_SEQ_SET(alignment->traceback, alignment->traceback_seq_len, TRACEBACK_MATCH);
      }
    }
    ++(alignment->traceback_seq_len);
    i += di;
    j += dj;
  }


  /* only build the displayed alignment if the verbosity level allows to display it */
  VERB_FILTER(VERBOSITY_HIGH, {
    int d = 0;
    int u;
    for (u = k - 1; u >= 0; --u) {
      alignment->to_display[d++] = ref[u];
    }
    alignment->to_display[d++] = '\n';
    for (u = k - 1; u >= 0; --u) {
      alignment->to_display[d++] = m[u];
    }
    alignment->to_display[d++] = '\n';
    for (u = k - 1; u >= 0; --u) {
      alignment->to_display[d++] = read[u];
    }
    alignment->to_display[d++] = '\n';
    alignment->to_display[d++] = '\0'; /* otherwise some leftovers from previous, longer alignments may show up... */
    free(read);
    free(ref);
    free(m);
  });
  return alignment->to_display;
}

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
                                   char* CIGAR_dest, char* SEQ_dest) {
  int i;
  int edit_distance = 0;
#ifdef NUCLEOTIDES
  int span = 0;
#else
  int span = 1; /* first is a match/subs letter */
#endif

  unsigned char prev_state = TRACEBACK_PAIR_MATCH;
  char* CIGAR_dest_pointer = CIGAR_dest;
  int SEQ_len = 0;
  int ref_idx = offset;

#ifndef NUCLEOTIDES
  CODE_TYPE ref_history = CODE_IDENTITY, read_history = CODE_IDENTITY;
  SEQ_dest[SEQ_len++] = CODE_IDENTITY;
#endif

  for (i = 0; i < traceback_len; ++i) {
    unsigned char traceback_cell      = TRACEBACK_SEQ_GET(traceback, i);
    unsigned char traceback_cell_type = GET_PAIR_TYPE(traceback_cell);
    unsigned char code                = traceback_cell & TRACEBACK_CODE_MASK;

    /* SEQUENCE */
    switch (traceback_cell_type) {

    case TRACEBACK_PAIR_MATCH:
      edit_distance--;
#ifdef NUCLEOTIDES
      code = NTH_CODE(ref_sequence, ref_idx);
#endif
    case TRACEBACK_PAIR_CODE_MISMATCH:
      edit_distance++;
#ifndef NUCLEOTIDES
      /*
       * retrieve color from reference
       * update the 'history' color of the reference
       */
      ref_history = COMPOSE(ref_history,  NTH_CODE(ref_sequence, ref_idx));
      /* compute the color to put in the read, and advance in the read:
       * it should be c such that:  read_history * c == ref_history
       * since each color is its own inverse, it follows that:  c = ref_history * read_history
       */
      code = COMPOSE(ref_history, read_history);
#endif
      SEQ_dest[SEQ_len++] = code;
#ifndef NUCLEOTIDES
      /* update the read history with the new color */
      read_history = COMPOSE(read_history, code);
#endif
      /* advance in the reference */
      ref_idx++;
      break;

#ifndef NUCLEOTIDES
    case TRACEBACK_PAIR_COLORBASE_MISMATCH:
      edit_distance++;
      /* copy color from read_color, and advance in both read and reference */
      SEQ_dest[SEQ_len++] = code;
      read_history = COMPOSE(read_history,  code);
      ref_history = COMPOSE(ref_history,  NTH_CODE(ref_sequence, ref_idx));
      ref_idx++;
      break;
#endif

    case TRACEBACK_PAIR_READ_INSERTION:
      edit_distance++;
      /* copy code from read and advance in the read */
      SEQ_dest[SEQ_len++] = code;
#ifndef NUCLEOTIDES
      read_history = COMPOSE(read_history,  code);
#endif
      break;

    case TRACEBACK_PAIR_READ_DELETION:
      edit_distance++;
      /* skip code from reference */
#ifndef NUCLEOTIDES
      ref_history = COMPOSE(ref_history,  NTH_CODE(ref_sequence, ref_idx));
#endif
      ref_idx++;
      break;

    default:
      break;
    }

    /* CIGAR like : I/D/M ...*/
    unsigned char crt_state = TRACEBACK_PAIR_MATCH;
    if ((traceback_cell_type == TRACEBACK_PAIR_READ_DELETION) || (traceback_cell_type == TRACEBACK_PAIR_READ_INSERTION)) {
      crt_state = traceback_cell_type;
    }

#define GET_CIGAR_FROM_PAIR_TYPE_SYMBOL(code) (((code) == TRACEBACK_PAIR_MATCH)               ? 'M' : \
                                               ((code) == TRACEBACK_PAIR_READ_INSERTION)      ? 'I' : \
                                               ((code) == TRACEBACK_PAIR_READ_DELETION)       ? 'D' : '?')

    if (crt_state == prev_state) {
      ++span;
    } else {
      if (span) {
        sprintf(CIGAR_dest_pointer, "%d%c", span, GET_CIGAR_FROM_PAIR_TYPE_SYMBOL(prev_state));
        CIGAR_dest_pointer = CIGAR_dest + strlen(CIGAR_dest);
      }
      span = 1;
      prev_state = crt_state;
    }
  }

  if (span) {
    sprintf(CIGAR_dest_pointer, "%d%c", span, GET_CIGAR_FROM_PAIR_TYPE_SYMBOL(prev_state));
  }

  /* translate obtained code (color/code) sequence into ascii using the reference letter into base letters */
#ifdef NUCLEOTIDES
  {
    int i;
    for (i = 0; i < SEQ_len; ++i) {
      SEQ_dest[i] = BASE_CODE_LETTER[(int)SEQ_dest[i]];
    }
  }
#else
  {
    CODE_TYPE code = ref_start_symbol;
    int i;
    for (i = 0; i < SEQ_len; ++i) {
      code = TRANSFORM(code, SEQ_dest[i]);
      SEQ_dest[i] = BASE_CODE_LETTER[code];
    }
  }
#endif
  return edit_distance;
}
