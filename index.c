#include <math.h>
#include "index.h"

#define IS_UNDEFINED(index_entry) ((index_entry) < 0)

#define COMPRESS 1

int GENERATE_KMER_HISTOGRAM = 0;
int FILTER_REPEATS = 1;
int ACCEPTED_STDEV_DISTANCE = DEFAULT_ACCEPTED_STDEV_DISTANCE ;
int KMER_HISTOGRAM_INTERVAL_WIDTH = 1;

 /**
 * Builds the index of a genome according to a given seed pattern
 * @param db
 * @param seed
 * @return The address of the created index
 */
IndexType* index__build_reference(ReferenceDBType* db, SeedType* seed) {
  IndexType* index;
  SAFE_FAILURE__ALLOC(index, 1, IndexType);
  /* The index size depends on the seed's weight : 4^w + 1*/
  index->index_size = (1 << (seed->weight << 1));// | 1;
  /* The map has the genome size minus the seed length*/
  index->map_size = db->size - seed->length + 1;
  if (index->map_size < 0) {
    index->map_size = 0;
    return index;
  }

  int *bookmark;
  int *counts = NULL;
  int *grouped_pos_lists;
  int count = 0, erased = 0;


  MaskedRegionType* crt_masked_region = db->masked_regions;

  index->db   = db;
  index->seed = seed;
  SAFE_FAILURE__ALLOC(index->first, index->index_size + 1, int);
  memset(index->first, 0xFF, (index->index_size + 1) * sizeof(int));
  index->first[index->index_size] = index->map_size;
  SAFE_FAILURE__ALLOC(bookmark,     index->index_size + 1, int);
  SAFE_FAILURE__ALLOC(index->map,   index->map_size,   int);
  SAFE_FAILURE__ALLOC(counts, index->index_size, int);
  memset(counts, 0x00, (index->index_size) * sizeof(int));

  SAFE_FAILURE__ALLOC(grouped_pos_lists, index->map_size+1, int);

  static char name[256];
  FILE* h_out = stderr;

  int max_occurrences = 0, intervals_nb = 0;
  int* intervals;

  if (GENERATE_KMER_HISTOGRAM) {
    snprintf(name, 255, "%s_l%d", db->name, seed->length);
    h_out = fopen(name, "w");
  }

  if (seed->length < db->size) {
    memset(index->map, 0xFF, index->map_size * sizeof(int));
    int i;
    for (i = 0; i < index->map_size; ++i) {
      while (crt_masked_region != NULL && (i + seed->length >= crt_masked_region->start)) {
        /* ignore this region, skip to the end */
        i = crt_masked_region->end-1; /* subtract 1 to compensate the ++i from the for loop */
        /* with this skip, we're out of the current masked region, go to the next */
        crt_masked_region = crt_masked_region->next;
        continue;
      }
#if (COMPRESS)
      int key = seed__apply_to_compressed(seed,
                                          db->sequence,
                                          i);
#else
      int key = seed__apply(seed,
                            db->sequence,
                            i);
#endif
      if (IS_UNDEFINED(index->first[key])) {
        index->first[key] = i;
      } else {
        index->map[bookmark[key]] = i;
      }
      bookmark[key] = i;

      /* count key occurrences */
      if (++counts[key] > max_occurrences) {
        max_occurrences = counts[key];
      }
      ++count;
      //VERB_FILTER(VERBOSITY_MODERATE, display_progress(i, index->map_size, 0));
    }
  }

  /* Generate intervals */
  if (GENERATE_KMER_HISTOGRAM) {
    intervals_nb = (max_occurrences + KMER_HISTOGRAM_INTERVAL_WIDTH) / KMER_HISTOGRAM_INTERVAL_WIDTH;
    SAFE_FAILURE__ALLOC(intervals, intervals_nb, int);
    int i;
    for (i = 0; i < index->index_size; ++i) {
      ++intervals[counts[i]/KMER_HISTOGRAM_INTERVAL_WIDTH];
    }
    for (i = 0; i < intervals_nb; ++i) {
#ifdef _OPENMP
#pragma omp ordered
#endif
      fprintf(h_out, "%d\t%d\n", MIN(intervals_nb, MIN(intervals_nb, (i + 1) * KMER_HISTOGRAM_INTERVAL_WIDTH - 1)), intervals[i]);
    }
    fclose(h_out);
    free(intervals);
  }

  /* Remove keys which are too frequent */
  if (FILTER_REPEATS) {
    int key;

    /* compute average and standard deviation of all kmers */
    double avg = 0;
    for (key = 0; key < index->index_size; ++key) {
      if (counts[key] > 0) {
        avg += 1.0 / count * counts[key] * counts[key];
      }
    }
    avg /= count;

    /* compute the std_dev "the hard way", to prevent from overflowing double */
    double std_dev = 0;
    for (key = 0; key < index->index_size; ++key) {
      if (counts[key] > 0) {
        std_dev += (counts[key] - avg) * (counts[key] - avg) / count;
      }
    }
    std_dev = sqrt(std_dev);

    /* "erase" from the index all keys that appear too often */
    char * seed_pattern = seed__to_string(seed);
    for (key = 0; key < index->index_size; ++key) {
      if (counts[key] && (counts[key] - avg) > ACCEPTED_STDEV_DISTANCE * std_dev) {
        index->first[key] = -1;
#ifdef _OPENMP
#pragma omp ordered
#endif
        VERB_FILTER(VERBOSITY_HIGH, WARNING__("%8d appearances of key 0x%x removed for seed \"%s\"", counts[key], key, seed_pattern));
        ++erased;
      }
    }
#ifdef _OPENMP
#pragma omp ordered
#endif
    VERB_FILTER(VERBOSITY_MODERATE, WARNING__("%8d key%s erased for seed \"%s\" (appearances > %d).", erased, erased == 1 ? "" : "s", seed_pattern, (int) (avg + ACCEPTED_STDEV_DISTANCE * std_dev + 0.9999)));
    free(seed_pattern);
  }

  /* Group the positions for the same key in the same memory chunk, to avoid cache failures when jumping to the next hit */
  /* Temporarily use bookmark for storing the rearranged index keys, and grouped_pos_lists for the position lists */
  /* Step 1: update start point for the position list of each key */

  bookmark[0] = 0;
  int key;
  for (key = 0; key < index->index_size; ++key) {
    bookmark[key + 1] = bookmark[key] + (index->first[key] >= 0 ? counts[key] : 0);
  }
  /* Step 2: fill in the map with the consecutive positions */
  int pos = bookmark[0];
  for (key = 0; key < index->index_size; ++key) {
    grouped_pos_lists[pos] = index->first[key];
    while (pos < bookmark[key + 1] && grouped_pos_lists[pos] >= 0) {
      grouped_pos_lists[pos + 1] = index->map[grouped_pos_lists[pos]];
      ++pos;
    }
  }

  /* Step 3: update pointers to first and map. This can be disabled with a compilation option (usually for timing tests) */
#ifndef DISABLE_CONTIG_INDEX
  {
    int* tmp     = index->first;
    index->first = bookmark;
    bookmark     = tmp;
  }
  {
    int* tmp          = index->map;
    index->map        = grouped_pos_lists;
    grouped_pos_lists = tmp;
  }
#endif
  /* Done making contiguous position lists */

  free(counts);
  free(grouped_pos_lists);
  free(bookmark);
  return index;
}


/**
 * Reset the index external "pointer" to the current hit
 */
inline void index__reset_extern_crt_hit(int * const hit_pointer) {
  *hit_pointer = -1;
}

/**
 * A hit in a reference index  (as field external to the index structure)
 * @param index
 * @param key
 * @return The position -- in the reference database -- of the subsequence
 * corresponding to the key
 */
inline int index__get_extern_next_hit(const IndexType *index, const int key, int * const hit_pointer) {
  register int a = *hit_pointer;

#ifdef DISABLE_CONTIG_INDEX

  if (a < 0) {
    a = index->first[key];
  } else {
    a = index->map[a];
  }
  *hit_pointer = a;
  return a;

#else /* ndef DISABLE_CONTIG_INDEX */

  if (a < 0) {
    a = index->first[key];
  } else {
    a++;
  }
  if (a >= index->first[key+1]){
    *hit_pointer = -1;
    return -1;
  }
  *hit_pointer = a;
  /*__builtin_prefetch(&(index->db->sequence[(index->map[(*hit_pointer)+5])>>2]));*/
  return a >= 0 ? index->map[a] : -1;

#endif /* ndef DISABLE_CONTIG_INDEX */
}

inline int index__get_extern_current_hit (const IndexType *index, const int key, const int *hit_pointer) {
#ifdef DISABLE_CONTIG_INDEX
  return *hit_pointer ;
#else
  return *hit_pointer >= 0 ? index->map[*hit_pointer] : -1;
#endif
}

/**
 * Destroy an indexed
 * @param index
 */
void index__destroy(IndexType* index) {
  free(index->first);
  index->first = NULL;
  free(index->map);
  index->map   = NULL;
}
