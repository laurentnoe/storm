#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "run.h"

int map_unordered = 0;

#include "stats.h"

#ifdef __AVX2__

#define MAX_MULTIPLE_HITS 8

const int simd_mul[INDEL_DATA_VECTOR_SIZE] = {8, 8, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2};
int(* const simd_fct_table[INDEL_DATA_VECTOR_SIZE])(unsigned char *, int *, unsigned char *) = {
  &alignment_avx2__align_octa,
  &alignment_avx2__align_octa,
  &alignment_avx2__align_octa,
  &alignment_avx2__align_octa,
  &alignment_avx2__align_quad,
  &alignment_avx2__align_quad,
  &alignment_avx2__align_quad,
  &alignment_avx2__align_quad,
  &alignment_avx2__align_pair,
  &alignment_avx2__align_pair,
  &alignment_avx2__align_pair,
  &alignment_avx2__align_pair,
  &alignment_avx2__align_pair,
  &alignment_avx2__align_pair,
  &alignment_avx2__align_pair,
  &alignment_avx2__align_pair
};
void(* const simd_init_fct_table[INDEL_DATA_VECTOR_SIZE])(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int) = {
  &alignment_avx2__init_octa,
  &alignment_avx2__init_octa,
  &alignment_avx2__init_octa,
  &alignment_avx2__init_octa,
  &alignment_avx2__init_quad,
  &alignment_avx2__init_quad,
  &alignment_avx2__init_quad,
  &alignment_avx2__init_quad,
  &alignment_avx2__init_pair,
  &alignment_avx2__init_pair,
  &alignment_avx2__init_pair,
  &alignment_avx2__init_pair,
  &alignment_avx2__init_pair,
  &alignment_avx2__init_pair,
  &alignment_avx2__init_pair,
  &alignment_avx2__init_pair
};
const int simd_N_BYTE_table[INDEL_DATA_VECTOR_SIZE] = {4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
void(* const simd_clean_fct)() = &alignment_avx2__clean;
#else
#ifdef __SSE2__

#define MAX_MULTIPLE_HITS 8

const int simd_mul[INDEL_DATA_VECTOR_SIZE] = {8, 8, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1};
int(* const simd_fct_table[INDEL_DATA_VECTOR_SIZE])(unsigned char *, int *, unsigned char *) = {
  &alignment_sse2__align_octa,
  &alignment_sse2__align_octa,
  &alignment_sse2__align_quad,
  &alignment_sse2__align_quad,
  &alignment_sse2__align_pair,
  &alignment_sse2__align_pair,
  &alignment_sse2__align_pair,
  &alignment_sse2__align_pair,
  &alignment_sse2__align_mono,
  &alignment_sse2__align_mono,
  &alignment_sse2__align_mono,
  &alignment_sse2__align_mono,
  &alignment_sse2__align_mono,
  &alignment_sse2__align_mono,
  &alignment_sse2__align_mono,
  &alignment_sse2__align_mono
};
void(* const simd_init_fct_table[INDEL_DATA_VECTOR_SIZE])(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int) = {
  &alignment_sse2__init_octa,
  &alignment_sse2__init_octa,
  &alignment_sse2__init_quad,
  &alignment_sse2__init_quad,
  &alignment_sse2__init_pair,
  &alignment_sse2__init_pair,
  &alignment_sse2__init_pair,
  &alignment_sse2__init_pair,
  &alignment_sse2__init_mono,
  &alignment_sse2__init_mono,
  &alignment_sse2__init_mono,
  &alignment_sse2__init_mono,
  &alignment_sse2__init_mono,
  &alignment_sse2__init_mono,
  &alignment_sse2__init_mono,
  &alignment_sse2__init_mono
};
const int simd_N_BYTE_table[INDEL_DATA_VECTOR_SIZE] = {2, 2, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
void(* const simd_clean_fct)() = &alignment_sse2__clean;
#else
#ifdef __SSE__

#define MAX_MULTIPLE_HITS 4

const int simd_mul[INDEL_DATA_VECTOR_SIZE] = {4, 4, 2, 2, 1, 1, 1, 1};
int(* const simd_fct_table[INDEL_DATA_VECTOR_SIZE])(unsigned char *, int *, unsigned char *) = {
  &alignment_sse__align_quad,
  &alignment_sse__align_quad,
  &alignment_sse__align_pair,
  &alignment_sse__align_pair,
  &alignment_sse__align_mono,
  &alignment_sse__align_mono,
  &alignment_sse__align_mono,
  &alignment_sse__align_mono
};
void(* const simd_init_fct_table[INDEL_DATA_VECTOR_SIZE])(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int) = {
  &alignment_sse__init_quad,
  &alignment_sse__init_quad,
  &alignment_sse__init_pair,
  &alignment_sse__init_pair,
  &alignment_sse__init_mono,
  &alignment_sse__init_mono,
  &alignment_sse__init_mono,
  &alignment_sse__init_mono
};
const int simd_N_BYTE_table[INDEL_DATA_VECTOR_SIZE] = {2, 2, 4, 4, 8, 8, 8, 8};
void(* const simd_clean_fct)() = &alignment_sse__clean;
#else

#warning "No __AVX2__, __SSE2__, __SSE__ defined : this program will be slow !!"

#define MAX_MULTIPLE_HITS 1

void fake_init (unsigned int match,
                unsigned int mismatch,
                unsigned int gapopen,
                unsigned int gapextends,
                unsigned int threshold,
                unsigned int readlength) {}
int fake_align (unsigned char * genome,
                int * pos_genome,
                unsigned char * read) { return 1; /* 0x01 */}
void fake_clean () {return; }
const int simd_mul[INDEL_DATA_VECTOR_SIZE] = {1, 1, 1, 1, 1, 1, 1, 1};
void(* const simd_init_fct_table[INDEL_DATA_VECTOR_SIZE])(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int) = {
  &fake_init,
  &fake_init,
  &fake_init,
  &fake_init,
  &fake_init,
  &fake_init,
  &fake_init,
  &fake_init
};
int(* const simd_fct_table[INDEL_DATA_VECTOR_SIZE])(unsigned char *, int *, unsigned char *) = {
  &fake_align,
  &fake_align,
  &fake_align,
  &fake_align,
  &fake_align,
  &fake_align,
  &fake_align,
  &fake_align
};
const int simd_N_BYTE_table[INDEL_DATA_VECTOR_SIZE] = {1, 1, 1, 1, 1, 1, 1, 1};
void(* const simd_clean_fct)() = &fake_clean;
#endif
#endif
#endif


int SHOW_TRACEBACK_PATTERNS = 0;
int LIST_UNMAPPED_READS = 0;


/*
 * #define __DONT__MAP__
 */

/**
 * Perform an alignment between two sequences
 * @param read_str First sequence (color code)
 * @param ref_str Second sequence (color code)
 * @param ref_masked_str Second sequence mask
 * @param read quality values for the first sequence
 * @param match
 * @param mismatch
 * @param gap_open
 * @param gap_extend
 * @param allowed_indels
 * @param output The output file descriptor
 */
int single_alignment(const char* read_str, const char* ref_str, const char* ref_masked_str, const char* qual_str,
                     const ScoreType match, const ScoreType mismatch, const ScoreType gap_open, const ScoreType gap_extend, const int allowed_indels, const int simd_allowed_diags,
                     FILE* output) {
  CODE_TYPE* read        = string_to_code(read_str);
  CODE_TYPE* ref         = string_to_code(ref_str);
  CODE_TYPE* ref_masked  = string_to_code(ref_masked_str);
  int read_len = strlen(read_str);
  short *quality = NULL;
  ScoreType best;
  AlignmentType* alignment;

  VERBOSITY = VERBOSITY_MODERATE;

  if (qual_str) {
    quality = parse_integer_sequence(qual_str, read_len);
  }

  alignment = alignment__create_single(read, quality, strlen(read_str), ref, ref_masked, strlen(ref_str), allowed_indels);
  alignment__set_params(alignment, match, mismatch, gap_open, gap_extend);

  best = alignment__align(alignment, MIN_SCORE);

  fprintf(output, "\n%s\n", alignment__traceback(alignment));
  fprintf(output, "Score: %d\n\n", best);
  alignment__destroy(alignment);
  free(read);
  free(ref);
  if (quality) {
    free(quality);
  }

  return best;
}



/*------------------------------------------------------------------------------------------------------
 * For multithreaded execution via openmp, some data structures need to be cloned for each thread
 */

#ifdef _OPENMP

/* Heap/Key/Hit Structures for each process (allocated per thread order, to avoid page conflicts between threads) */
/* [FIXME] must allocate alignment here to ensure that pagging will be "malloc contiguous" for each thread ... and moreover "page" distant between two threads */
#define HEAP_KEY_HIT_REVSEQ__DECLARE HitType ** heaps = NULL; int *** keys = NULL; int *** hit_pointers = NULL; ReadDataType * tmp_sequence = NULL;

#define HEAP_KEY_HIT_REVSEQ__ALLOC(__heap_size__,__nb_seeds__,__read_len__) {  \
    int t;                                                                     \
    SAFE_FAILURE__ALLOC(heaps, MAX_THREADS, HitType*);                         \
    SAFE_FAILURE__ALLOC(keys, MAX_THREADS, int**);                             \
    SAFE_FAILURE__ALLOC(hit_pointers, MAX_THREADS, int**);                     \
    SAFE_FAILURE__ALLOC(tmp_sequence, MAX_THREADS, ReadDataType);              \
    for (t = 0; t < MAX_THREADS; ++t) {                                        \
      SAFE_FAILURE__ALLOC(heaps[t], __heap_size__, HitType);                   \
      SAFE_FAILURE__ALLOC(keys[t], __nb_seeds__, int *);                       \
      SAFE_FAILURE__ALLOC(hit_pointers[t], __nb_seeds__, int *);               \
      SAFE_FAILURE__ALLOC(tmp_sequence[t].sequence, __read_len__, CODE_TYPE);  \
      tmp_sequence[t].quality = NULL;                                          \
      int u;                                                                   \
      for (u = 0; u < __nb_seeds__; ++u) {                                     \
        SAFE_FAILURE__ALLOC(keys[t][u], __read_len__, int);                    \
        SAFE_FAILURE__ALLOC(hit_pointers[t][u], __read_len__, int);            \
      }                                                                        \
    }                                                                          \
  }

#define KEY_HIT__RESET(__nb_seeds__,__read_len__) {                     \
    int u;                                                              \
    for (u = 0; u < __nb_seeds__; ++u) {                                \
      memset(key[u], '\xff', __read_len__*sizeof(int));                 \
      memset(hit_pointer[u], '\xff', __read_len__*sizeof(int));         \
    }                                                                   \
  }

#define HEAP__REF        (heaps[_ogtn])
#define KEY__REF         (keys[_ogtn])
#define HIT_POINTER__REF (hit_pointers[_ogtn])

#define HEAP_KEY_HIT_REVSEQ__DESTROY(__nb_seeds__) {   \
    int t;                                             \
    for (t = 0; t < MAX_THREADS; ++t) {                \
      int u;                                           \
      for (u = 0; u < __nb_seeds__; ++u) {             \
        free(keys[t][u]);                              \
        free(hit_pointers[t][u]);                      \
      }                                                \
      free(heaps[t]);                                  \
      free(keys[t]);                                   \
      free(hit_pointers[t]);                           \
      free(tmp_sequence[t].sequence);                  \
      if (tmp_sequence[t].quality) {                   \
        free(tmp_sequence[t].quality);                 \
      }                                                \
    }                                                  \
    free(heaps);                                       \
    free(keys);                                        \
    free(hit_pointers);                                \
    free(tmp_sequence);                                \
  }


/* Alignment Structures for each process */
#define ALIGNMENT__DECLARE AlignmentType* alignment = NULL;

#define ALIGNMENT__INIT(read_len, ref_len, allowed_indels, match, mismatch, gap_open, gap_extend)  { \
    int t;                                                                                           \
    SAFE_FAILURE__ALLOC(alignment, MAX_THREADS, AlignmentType);                                      \
    for (t = 0; t < MAX_THREADS; ++t) {                                                              \
      alignment__init(&alignment[t], read_len, ref_len, allowed_indels);                             \
      alignment__set_params(&alignment[t], match, mismatch, gap_open, gap_extend);                   \
    }                                                                                                \
  }

#define ALIGNMENT__RESET_READ(read, quality) {                              \
    alignment__reset_with_compressed_read(&alignment[_ogtn], read, quality); \
  }

#define ALIGNMENT__REF &alignment[_ogtn]

#define ALIGNMENT__DESTROY {              \
    int t;                                \
    for (t = 0; t < MAX_THREADS; ++t) {   \
      alignment__destroy(&alignment[t]);  \
    }                                     \
    free(alignment);                      \
  }

/* Reverse complementary data for each process */
#define REV_SEQ__REF(tmp_sequence)     &tmp_sequence[_ogtn]

#define REV_SEQ__SEQ_REF(tmp_sequence)  tmp_sequence[_ogtn].sequence

#define REV_SEQ__QUAL_REF(tmp_sequence) tmp_sequence[_ogtn].quality


#else

/* Heap/Key/Hit Structures for each process */
#define HEAP_KEY_HIT_REVSEQ__DECLARE HitType * heap = NULL; int ** key = NULL;  int ** hit_pointer = NULL; ReadDataType tmp_sequence = {0};

#define HEAP_KEY_HIT_REVSEQ__ALLOC(__heap_size__,__nb_seeds__,__read_len__) {  \
    SAFE_FAILURE__ALLOC(heap, __heap_size__, HitType);                         \
    SAFE_FAILURE__ALLOC(key, __nb_seeds__, int *);                             \
    SAFE_FAILURE__ALLOC(hit_pointer, __nb_seeds__, int *);                     \
    SAFE_FAILURE__ALLOC(tmp_sequence.sequence, __read_len__, CODE_TYPE);       \
    tmp_sequence.quality = NULL;                                               \
    int u;                                                                     \
    for (u = 0; u < __nb_seeds__; ++u) {                                       \
      SAFE_FAILURE__ALLOC(key[u], __read_len__, int);                          \
      SAFE_FAILURE__ALLOC(hit_pointer[u], __read_len__, int);                  \
    }                                                                          \
  }

#define KEY_HIT__RESET(__nb_seeds__,__read_len__) {                     \
    int u;                                                              \
    for (u = 0; u < __nb_seeds__; ++u) {                                \
      memset(key[u], '\xff', __read_len__*sizeof(int));                 \
      memset(hit_pointer[u], '\xff', __read_len__*sizeof(int));         \
    }                                                                   \
  }

#define HEAP__REF        (heap)
#define KEY__REF         (key)
#define HIT_POINTER__REF (hit_pointer)

#define HEAP_KEY_HIT_REVSEQ__DESTROY(__nb_seeds__) {  \
    int u;                                            \
    for (u = 0; u < __nb_seeds__; ++u) {              \
      free(key[u]);                                   \
      free(hit_pointer[u]);                           \
    }                                                 \
    free(heap);                                       \
    free(key);                                        \
    free(hit_pointer);                                \
    free(tmp_sequence.sequence);                      \
    if (tmp_sequence.quality) {                       \
      free(tmp_sequence.quality);                     \
    }                                                 \
  }

/* Alignment Structure */
#define ALIGNMENT__DECLARE  AlignmentType alignment = {0};

#define ALIGNMENT__INIT(read_len, ref_len, allowed_indels, match, mismatch, gap_open, gap_extend) { \
    alignment__init(&alignment, read_len, ref_len, allowed_indels);                                 \
    alignment__set_params(&alignment, match, mismatch, gap_open, gap_extend);                       \
  }

#define ALIGNMENT__RESET_READ(read, quality) {                          \
    alignment__reset_with_compressed_read(&alignment, read, quality);   \
  }

#define ALIGNMENT__REF &alignment

#define ALIGNMENT__DESTROY alignment__destroy(&alignment);

/* Reverse complementary data */
#define REV_SEQ__REF(tmp_sequence)  &tmp_sequence

#define REV_SEQ__SEQ_REF(tmp_sequence)  tmp_sequence.sequence

#define REV_SEQ__QUAL_REF(tmp_sequence) tmp_sequence.quality

#endif



/*---------------------------------------------->8--------------------------------------------------------*/

/* Heap structure for sorting hits */
typedef struct HitType {
  int   hit_pos;
  short read_pos;
  short seed_id;
} __attribute__((__packed__)) HitType;

/* insert at position $__size__$ then sieve from bottom to top (increment $__size__$) */
#define INSERT_HEAP(__heap__,__size__,__hit_pos__,__read_pos__,__seed_id__) \
  {                                                                         \
    int __u__ = __size__;                                                   \
    int __t__ = (__u__-1)>>1;                                               \
    while (                                                                 \
            (__u__ > 0)                                                     \
            &&                                                              \
            (__hit_pos__ <  (__heap__)[__t__].hit_pos)                      \
          ) {                                                               \
       (__heap__)[__u__] = (__heap__)[__t__];                               \
       __u__ = __t__;                                                       \
       __t__ = (__u__-1)>>1;                                                \
    }                                                                       \
    (__heap__)[__u__].hit_pos  = __hit_pos__  ;                             \
    (__heap__)[__u__].read_pos = __read_pos__ ;                             \
    (__heap__)[__u__].seed_id  = __seed_id__  ;                             \
    __size__ ++;                                                            \
  }

/* replace at position $0$ (erase previous element) then sieve from top to bottom */
#define REP_TOP_SIEVE_HEAP(__heap__,__size__,__hit_pos__,__read_pos__,__seed_id__)    \
  {                                                                                   \
    int __u__ = 0;                                                                    \
    while (__u__ < __size__) {                                                        \
       int __u_min__ = __u__;                                                         \
       int __min__   = __hit_pos__;                                                   \
       if ( 2*__u__+1 < __size__) {                                                   \
         if ((__heap__)[2*__u__+1].hit_pos < __min__) {                               \
           __u_min__ = 2*__u__+1;                                                     \
           __min__   = (__heap__)[2*__u__+1].hit_pos;                                 \
         }                                                                            \
         if ( 2*__u__+2 < __size__) {                                                 \
           if ((__heap__)[2*__u__+2].hit_pos < __min__) {                             \
             __u_min__ = 2*__u__+2;                                                   \
           }                                                                          \
         }                                                                            \
       }                                                                              \
       if (__u_min__ == __u__)                                                        \
         break;                                                                       \
       (__heap__)[__u__] = (__heap__)[__u_min__];                                     \
       __u__ = __u_min__;                                                             \
    }                                                                                 \
    (__heap__)[__u__].hit_pos  = __hit_pos__  ;                                       \
    (__heap__)[__u__].read_pos = __read_pos__ ;                                       \
    (__heap__)[__u__].seed_id  = __seed_id__  ;                                       \
  }


/* delete at position $0$ (by using the last element in the sieve) then sieve from top to bottom */
#define DEL_TOP_SIEVE_HEAP(__heap__,__size__)                                         \
  {                                                                                   \
    __size__ -- ;                                                                     \
    int __hit_pos__  = (__heap__)[__size__].hit_pos;                                  \
    int __read_pos__ = (__heap__)[__size__].read_pos;                                 \
    int __seed_id__  = (__heap__)[__size__].seed_id;                                  \
    REP_TOP_SIEVE_HEAP(__heap__,__size__,__hit_pos__,__read_pos__,__seed_id__)        \
  }




/**
 * Find and register in a hit_map all the extended hits found for one given read
 * This part is the most time consuming in SToRM
 * @param read_seq The read's sequence to be processed
 * @param read_len The read's sequence length
 * @param read_id  The read's index (for updating the map only)
 * @param ref_id The reference's index (for updating the map only)
 * @param seeds_count The number of seeds available
 * @param indexes An array of indexes of the current ref (one per seed)
 * @param alignment The alignment data structure
 * @param map the read map to update with good alignments
 * @param number of diagonal allowed for the simd code (for selection the right simd function)
 * @param alignement done in forward/reverse (sequence already reversed, for updating the map only)
 */

static inline void process_read(
                                HitType * heap, int ** key, int ** hit_pointer,
                                CODE_TYPE* read_seq, const int read_len, const int read_id,
                                const int ref_id, int seeds_count,
                                IndexType** indexes,
                                AlignmentType* alignment, HitMapType* map,
                                const int simd_allowed_diags,
                                const int alignment_sense) {

  int I                      = alignment->params.allowed_indels;

  CODE_TYPE* ref_seq         = ((ReferenceDBType*)indexes[0]->db)->sequence;
  CODE_TYPE* ref_seq_masked  = ((ReferenceDBType*)indexes[0]->db)->sequence_masked;
  ScoreType score_threshold  = min_accepted_score;


  /* reset key and hit_pointer */
  KEY_HIT__RESET(seeds_count,read_len);

  /*
   * (1) first fill heap : at most one element (hit) in the heap per (read_pos,seed_id)
   */
  int heap_size = 0;
  int si;
  for (si = 0; si < seeds_count; ++si) {
    if (indexes[si]->seed->positions) {
      int k;
      for (k = 0; k < indexes[si]->seed->positions_count; ++k) {
        int read_pos = indexes[si]->seed->positions[k];
        key[si][read_pos] = seed__apply_to_compressed(indexes[si]->seed, read_seq, read_pos);
        int ref_pos       = index__get_extern_next_hit(indexes[si], key[si][read_pos], hit_pointer[si]+read_pos);
        if (ref_pos >= 0) {
          INSERT_HEAP(heap,heap_size,(ref_pos-read_pos),read_pos,si);
        }
      }
    } else {
      int read_pos;
      for (read_pos = 0; read_pos < read_len - indexes[si]->seed->length + 1; ++read_pos) {
        key[si][read_pos] = seed__apply_to_compressed(indexes[si]->seed, read_seq, read_pos);
        int ref_pos       = index__get_extern_next_hit(indexes[si], key[si][read_pos], hit_pointer[si]+read_pos);
        if (ref_pos >= 0) {
          INSERT_HEAP(heap,heap_size,(ref_pos-read_pos),read_pos,si);
        }
      }
    }
  }


  /*
   * (2) then find the smallest hit, update the simd search array, and run the simd search when array is full
   */
  int old_hit_pos    = -I;
  int hits_pos_index = 0;
  int hits_pos[MAX_MULTIPLE_HITS] = {0};
  int simd_win_start[MAX_MULTIPLE_HITS] = {0};

  while (heap_size) {
    int   hit_pos  = heap[0].hit_pos;
    short read_pos = heap[0].read_pos;
    short seed_id  = heap[0].seed_id;

    /* heap update can be done in parallel with ...*/
    int new_ref_pos = index__get_extern_next_hit(indexes[seed_id], key[seed_id][read_pos], hit_pointer[seed_id]+read_pos);

    if (new_ref_pos >= 0) {
      REP_TOP_SIEVE_HEAP(heap,heap_size,(new_ref_pos-read_pos),read_pos,seed_id);
    } else {
      DEL_TOP_SIEVE_HEAP(heap,heap_size);
    }

    /* ... managing hits and alignments */
    if (hit_pos > old_hit_pos) {
      hits_pos[hits_pos_index] = hit_pos;
      simd_win_start[hits_pos_index] = hit_pos - simd_allowed_diags;
      hits_pos_index++; hits_pos_index %= simd_mul[simd_allowed_diags];
      old_hit_pos = hit_pos;

      VERB_FILTER(VERBOSITY_ANNOYING, INFO__(("\nRead:%d@[%d]: HIT Reference:%d@[%d]", read_id, read_pos, ref_id, hit_pos)););

      if (hits_pos_index == 0) {
        /* alignment with the simd_mul[simd_allowed_diags] hits found */

        /* fast filtering using the SIMD alignment functions */
        int x = simd_fct_table[simd_allowed_diags](ref_seq, simd_win_start, read_seq);
        int p = 0;

        /* check which alignment has a hit (if any, thus when x != 0)*/
        while (x != 0) {
          if (x & 1) {

            int ref_window = MAX(hits_pos[p] - I, -simd_allowed_diags);

            VERB_FILTER(VERBOSITY_ANNOYING, INFO__(("\nRead:%d[%d] PASSED SIMD FILTER Reference:%d@[%d]", read_id, read_pos, ref_id, ref_window)););

            alignment__reset_with_compressed_ref(alignment, ref_seq, ref_seq_masked, ref_window);

            /* if the hit is not in a region that has already been checked, and the alignment score is acceptable, memorize it */
            ScoreType score;
            if ((score = alignment__align(alignment, score_threshold)) > score_threshold) {

              alignment__traceback(alignment);

              /* display the alignment */
              VERB_FILTER(VERBOSITY_HIGH, INFO__(("\nReference:%d@[%d-%d], Read:%d@[%d–%d]\n%sScore:%d\n", ref_id, ref_window + alignment->best_j0, ref_window + alignment->best_j, read_id,  alignment->best_i0,  alignment->best_i, alignment->to_display, score)););
              /* memorize in a map */
              hit_map__update(map, read_id, ref_id, ref_window, alignment, alignment_sense);
              if (map->map[read_id][MAP_DETAIL_SIZE-1].score > score_threshold) {
                score_threshold = map->map[read_id][MAP_DETAIL_SIZE-1].score;
              }
            }
          }
          /* next alignment in the SIMD result (binary flag output)*/
          x >>= 1;
          ++p;
        }
      }
    }
  }


  /*
   * (3) last hits kept in the simd search array must be processed too ...
   */

  if (hits_pos_index) {
    /* fill the vector */
    while (hits_pos_index) {
            hits_pos[hits_pos_index] =       hits_pos[hits_pos_index-1];
      simd_win_start[hits_pos_index] = simd_win_start[hits_pos_index-1];
      hits_pos_index++; hits_pos_index %= simd_mul[simd_allowed_diags];
    }

    /* fast filtering using the SIMD alignment functions */
    int x = simd_fct_table[simd_allowed_diags](ref_seq, simd_win_start, read_seq);
    int p = 0;

    /* check which alignment has a hit (if any, thus when x != 0)*/
    while (x != 0) {
      if (x & 1) {

        int ref_window = MAX(hits_pos[p] - I, -simd_allowed_diags);

        VERB_FILTER(VERBOSITY_ANNOYING, INFO__(("\nRead %d PASSED SIMD FILTER Reference@%d", read_id, ref_window)););

        alignment__reset_with_compressed_ref(alignment, ref_seq, ref_seq_masked, ref_window);

        /* if the hit is not in a region that has already been checked, and the alignment score is acceptable, memorize it */
        ScoreType score;
        if ((score = alignment__align(alignment, score_threshold)) > score_threshold) {

          alignment__traceback(alignment);

          /* display the alignment */
          VERB_FILTER(VERBOSITY_HIGH, INFO__(("\nReference@%d, Read %d\n%sScore: %d\n", ref_window + alignment->best_j0, read_id, alignment->to_display, score)););
          /* memorize in a map */
          hit_map__update(map, read_id, ref_id, ref_window, alignment, alignment_sense);
          if (map->map[read_id][MAP_DETAIL_SIZE-1].score > score_threshold) {
            score_threshold = map->map[read_id][MAP_DETAIL_SIZE-1].score;
          }
        }
      }
      /* next alignment in the SIMD result (binary flag output)*/
      x >>= 1;
      ++p;
    }
  }
}


/**
 * Reads mapped on a reference
 * @param reads_filename
 * @param qual_filename
 * @param ref_filename
 * @param seed_list
 * @param match
 * @param mismatch
 * @param gap_open
 * @param gap_extend
 * @param allowd_indels
 * @param output
 * @param unmapped_FASTQ_output
 */
int reads_against_references(const char* reads_filename, const char* qual_filename, const char* ref_filename,
                             const char* seedslist,
                             const ScoreType match, const ScoreType mismatch, const ScoreType gap_open, const ScoreType gap_extend, const int allowed_indels, const int simd_allowed_diags,
                             FILE* output, FILE* unmapped_FASTQ_output) {
  ReadsDBType       reads_db;
  ReferenceDBType*  ref_dbs;
  int               ref_dbs_size;
  IndexType**       ref_index;
  SeedType **       seeds;


#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(MAX_THREADS);
#endif

  ALIGNMENT__DECLARE;

  HEAP_KEY_HIT_REVSEQ__DECLARE;

  HitMapType* map;

  time_t start_time = time(NULL), crt_time, crt_time1;

  /* global variable used to allocate the correct number of byte after each ALLOC, in order to enable block reading without alarming valgrind */
  N_BYTES = simd_N_BYTE_table[simd_allowed_diags];

  /* create seeds */
  int seeds_count = seed__parse_list(seedslist, &seeds);
  /* verify seed validity */
  if (seeds_count == RETURN_INPUT_ERR) {
    ERROR__(("The seed pattern is invalid: \"%s\".\nExpected: {0,1}+ or {-,#}+ with at least one '1'/'#' (seed separator ';').", seedslist));
    exit (RETURN_INPUT_ERR);
  } else if (seeds_count <= 0) {
    ERROR__(("The seed input is invalid: \"%s\".\nPlease provide at least one seed.", seedslist));
    exit (RETURN_INPUT_ERR);
  }
  VERB_FILTER(VERBOSITY_MODERATE,
            INFO__(("The following %d seed%s will be used:\n", seeds_count, (seeds_count > 1 ? "s" : "")));
            int si; for (si = 0; si < seeds_count; ++si) seed__display(seeds[si]);
            printf("\n");
            );

  /* load the two databases */
  /* load the reads */
  VERB_FILTER(VERBOSITY_MODERATE, printf("Loading reads database...\n"););
  crt_time = time(NULL);
  if (load_reads_db(reads_filename, qual_filename, &reads_db) <= 0) {
    ERROR__(("Reads could not be loaded. Exiting."));
    exit (RETURN_INPUT_ERR);
  }
  map = hit_map__create(reads_db.size, allowed_indels);
  VERB_FILTER(VERBOSITY_MODERATE, printf("Loaded %ld reads (read length %d) in %ld seconds.\n\n", reads_db.size, reads_db.read_len, time(NULL) - crt_time););

  /* sort reads in alphanumeric order */
  VERB_FILTER(VERBOSITY_MODERATE, printf("Sorting reads database...\n"););
  crt_time = time(NULL);
  sort_reads_db(&reads_db);
  VERB_FILTER(VERBOSITY_MODERATE, printf("Sorted %ld reads in %ld seconds.\n\n", reads_db.size, time(NULL) - crt_time););

  /* load the reference */
  VERB_FILTER(VERBOSITY_MODERATE, printf("Loading reference database...\n"););
  crt_time = time(NULL);
  if ((ref_dbs_size = load_reference_db(ref_filename, reads_db.read_len, &ref_dbs)) <= 0) {
    ERROR__(("Reference could not be loaded. Exiting."));
    exit(RETURN_INPUT_ERR);
  }
  VERB_FILTER(VERBOSITY_MODERATE, printf("Loaded %d reference sequence%s in %ld seconds.\n\n", ref_dbs_size, (ref_dbs_size == 1 ? "" : "s"), time(NULL) - crt_time););

  /* SIMD filter initialization */
  simd_init_fct_table[simd_allowed_diags](abs(match), abs(mismatch), abs(gap_open), abs(gap_extend), min_accepted_score_simd_filter, reads_db.read_len);

  /* compute heap size needed */
  int heap_size = 0;
  int si;
  for (si = 0; si < seeds_count; ++si) {
    if (seeds[si]->positions)
      heap_size += seeds[si]->positions_count;
    else
      heap_size += reads_db.read_len;/* enought : used to keep distances ...*/
  }

  /* create indexes */
  SAFE_FAILURE__ALLOC(ref_index, seeds_count, IndexType*);
  /* create alignment data */
  int ref_window_len = reads_db.read_len + (allowed_indels<<1);
  ALIGNMENT__INIT(reads_db.read_len, ref_window_len, allowed_indels, match, mismatch, gap_open, gap_extend);
  /* create Heaps,Key,Hits,Revseq used by threads */
  HEAP_KEY_HIT_REVSEQ__ALLOC(heap_size,seeds_count,reads_db.read_len);

  crt_time = time(NULL);

  /* process each reference sequence */
  int ref_id;
  for (ref_id = 0; ref_id < ref_dbs_size; ++ref_id) {
    ReferenceDBType* ref_db = &ref_dbs[ref_id];
#ifdef NUCLEOTIDES
    VERB_FILTER(VERBOSITY_MODERATE, INFO__(("\n\nProcessing reference sequence %s (%d of %d, %d bases)...\n", ref_db->name?ref_db->name:"(null)", ref_id + 1, ref_dbs_size, ref_db->size)););
#else
    VERB_FILTER(VERBOSITY_MODERATE, INFO__(("\n\nProcessing reference sequence %s (%d of %d, %d colors)...\n", ref_db->name?ref_db->name:"(null)", ref_id + 1, ref_dbs_size, ref_db->size)););
#endif

    /* create index */
    VERB_FILTER(VERBOSITY_MODERATE, printf("Creating reference index...\n"););
    crt_time1 = time(NULL);
    int si;
#ifdef _OPENMP
#pragma omp parallel for ordered shared(ref_index, ref_window_len, ref_db, seeds)
#endif
    for (si = 0; si < seeds_count; ++si) {
      ref_index[si] = index__build_reference(ref_db, seeds[si]);
    }
    VERB_FILTER(VERBOSITY_MODERATE, printf("Created %d indexes in %ld seconds.\n\n", seeds_count, time(NULL) - crt_time1););


    /* on each read, pass each seed, find the hits, and align */
    VERB_FILTER(VERBOSITY_MODERATE, printf("Start search...\n"););

    crt_time1 = time(NULL);
    if (ALIGNMENT_SENSE & ALIGNMENT_SENSE_FORWARD) {
      VERB_FILTER(VERBOSITY_MODERATE, printf("Forward...\n"););

      int p_read_id = 0;
      int read_id;

      VERB_FILTER(VERBOSITY_MODERATE, display_progress(p_read_id, reads_db.size, map->mapped););

#ifdef _OPENMP
#pragma omp parallel for shared(p_read_id, reads_db, ref_index, seeds) schedule(dynamic,256)
#endif
      for (read_id = 0; read_id < reads_db.size; ++read_id) {

#ifdef _OPENMP
        const int _ogtn = omp_get_thread_num();
#endif

        /* set the read */
        ALIGNMENT__RESET_READ(reads_db.reads[read_id].sequence, reads_db.reads[read_id].quality);

        process_read(HEAP__REF, KEY__REF, HIT_POINTER__REF,
                     (reads_db.reads[read_id]).sequence, reads_db.read_len, read_id,
                     ref_id,  seeds_count,
                     ref_index,
                     ALIGNMENT__REF, map,
                     simd_allowed_diags, ALIGNMENT_SENSE_FORWARD
                 );

        /* do not display this progress bar if other details (such as specific local alignments) are also displayed. */
        VERB_FILTER(VERBOSITY_MODERATE,
                    if ( !(p_read_id & 0x3ff) ) {
                      display_progress(p_read_id, reads_db.size, map->mapped);
                    }
                    );
#ifdef _OPENMP
#pragma omp atomic
#endif
        p_read_id++;

      }

      VERB_FILTER(VERBOSITY_MODERATE, display_progress(reads_db.size, reads_db.size, map->mapped););
      VERB_FILTER(VERBOSITY_NONE, printf("Forward completed in %ld seconds, %d of %d reads aligned.\n\n", time(NULL) - crt_time1, map->mapped, map->size));
    }



    crt_time1 = time(NULL);
    if (ALIGNMENT_SENSE & ALIGNMENT_SENSE_REVERSE) {
      VERB_FILTER(VERBOSITY_MODERATE, printf("Reverse complementary...\n"););

      int p_read_id = 0;
      int read_id;

      VERB_FILTER(VERBOSITY_MODERATE, display_progress(p_read_id, reads_db.size, map->mapped));

#ifdef _OPENMP
#pragma omp parallel for shared(p_read_id, reads_db, ref_index, seeds) schedule(dynamic,256)
#endif
      for (read_id = 0; read_id < reads_db.size; ++read_id) {

#ifdef _OPENMP
        const int _ogtn = omp_get_thread_num();
#endif

        /* set the read */
        read__reverse(&reads_db.reads[read_id], REV_SEQ__REF(tmp_sequence), reads_db.read_len);
        ALIGNMENT__RESET_READ(REV_SEQ__SEQ_REF(tmp_sequence), REV_SEQ__QUAL_REF(tmp_sequence));

        process_read(HEAP__REF, KEY__REF, HIT_POINTER__REF,
                     REV_SEQ__SEQ_REF(tmp_sequence), reads_db.read_len, read_id,
                     ref_id,  seeds_count,
                     ref_index,
                     ALIGNMENT__REF, map,
                     simd_allowed_diags, ALIGNMENT_SENSE_REVERSE
                 );

        /* do not display this progress bar if other details (such as specific local alignments) are also displayed. */
        VERB_FILTER(VERBOSITY_MODERATE,
                    if ( !(p_read_id & 0x3ff) ) {
                      display_progress(p_read_id, reads_db.size, map->mapped);
                    }
                    );
#ifdef _OPENMP
#pragma omp atomic
#endif
        p_read_id++;

      }

      VERB_FILTER(VERBOSITY_MODERATE, display_progress(reads_db.size, reads_db.size, map->mapped));
      VERB_FILTER(VERBOSITY_NONE, printf("Reverse complementary completed in %ld seconds, %d of %d reads aligned.\n\n", time(NULL) - crt_time1, map->mapped, map->size));
    }

    /* clear the index for each seed of the current reference */
    {
      int si;
      for (si = 0; si < seeds_count; ++si) {
        index__destroy(ref_index[si]);
      free(ref_index[si]);
      ref_index[si] = NULL;
      }
    }
  } /* for (ref_id = 0; ref_id < ref_counts; ++ref_id) */

  /* clear SIMD allocated data */
  simd_clean_fct();

  /* clear the seeds */
  {
    int si;
    for (si = 0; si < seeds_count; ++si) {
      seed__destroy(seeds[si]);
      free(seeds[si]);
      seeds[si] = NULL;
    }
  }
  /* erase the arrays allocated for pointing indexes and seeds */
  free(ref_index);
  ref_index = NULL;
  free(seeds);
  seeds = NULL;

  /* erase alignments,heaps,revseq ... */
  ALIGNMENT__DESTROY;
  HEAP_KEY_HIT_REVSEQ__DESTROY(seeds_count);

  VERB_FILTER(VERBOSITY_NONE, printf("\nAligned %d of %d reads (%5.2f%%) in %ld seconds.\n", map->mapped, map->size, (map->mapped*100.0/map->size), time(NULL) - crt_time));


#ifndef __DONT__MAP__

  /* generate mapping */
  if (map_unordered == 0) {

    crt_time = time(NULL);

    VERB_FILTER(VERBOSITY_MODERATE, printf("Reference mapping...\n"););
    GenomeMapType*  genome_map = genome_map__create(map, ref_dbs, ref_dbs_size, &reads_db);
    genome_map__build(genome_map);

    /* it may be useful to see what the alignments look like */
    if (SHOW_TRACEBACK_PATTERNS) {
      display_tracebacks(genome_map);
    }

    /* it may be useful to see which reads didn't make it */
    if (LIST_UNMAPPED_READS) {
      list_unmapped_reads(genome_map);
      list_unmapped_reads_translated(genome_map);
    }

    /* generate output */
    VERB_FILTER(VERBOSITY_NONE, printf("\nMap built in %ld seconds. Generating SAM output...\n", time(NULL) - crt_time););
    genome_map__generate_SAM_output(genome_map, output);

    /* erase the genome_map */
    genome_map__destroy(genome_map);
    free(genome_map);
    genome_map = NULL;
  } else {

    /* generate output */
    VERB_FILTER(VERBOSITY_NONE, printf("\nGenerating SAM output...\n"););
    hit_map__generate_SAM_output(map,
                                 &reads_db,
                                 ref_dbs, ref_dbs_size,
                                 output, map_unordered);

  }
#endif /* __DONT__MAP__ */

  /* output unmmapped reads */
  if (unmapped_FASTQ_output) {
     VERB_FILTER(VERBOSITY_NONE, printf("\nGenerating FastQ output for unmapped reads...\n"););
     hit_map__generate_unmapped_FASTQ_output(map,
                                             &reads_db,
                                             unmapped_FASTQ_output);
  }
  /* clear the hit_map */
  hit_map__destroy(map);
  free(map);
  map = NULL;

  /* clear the references [FIXME] can done before ? realignment problem on the ref if (map_unordered == 0) */
  {
    int ref_id;
    for (ref_id = 0; ref_id < ref_dbs_size; ++ref_id) {
      ReferenceDBType* ref_db = &ref_dbs[ref_id];
      clear_reference_db(ref_db);
    }
  }

  /* clear the reads [FIXME] can be done before ? previous problem + quality problem */
  clear_reads_db(&reads_db);
  free(ref_dbs);
  ref_dbs = NULL;

  VERB_FILTER(VERBOSITY_MODERATE, INFO__(("\nAll done in %ld seconds.\n", time(NULL) - start_time)););
  return RETURN_SUCCESS;
}
