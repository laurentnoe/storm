#ifndef _INDEX_H_
#define _INDEX_H_

#ifdef _OPENMP
#include <omp.h>
#endif

#include <math.h>

#include "seed.h"
#include "load_data.h"
#include "reads_data.h"
#include "reference_data.h"


#define DEFAULT_ACCEPTED_STDEV_DISTANCE 100

/**
 * @struct IndexType
 * Index for one seed
 */
typedef struct IndexType {
  /** reference to the genome sequence db */
  ReferenceDBType *db;
  /** reference to the seed used to build the index */
  SeedType* seed;
  /** "seedcode" indexed array : gives the first "position" of
   *  a given "seedcode"
   */
  int* first;
  /** "position" indexed array : gives the next "position" of
   *   a "seedcode" that also occurs at this "position"
   *   (linked list principle)
   *   @see first
   */
  int* map;
  /** size of the "first" array */
  int index_size;
  /** size of the "map" array */
  int map_size;
} IndexType;

/**
 * Builds the index of a genome according to a given seed pattern
 * @param db
 * @param seed
 * @return The address of the created index
 */
IndexType* index__build_reference(ReferenceDBType* db, SeedType* seed);

/**
 * Reset the index external "pointer" to the current hit
 */
void index__reset_extern_crt_hit(int *hit_pointer);

/**
 * The next hit in a reference index (as field external to the index structure)
 * @param index
 * @param key
 * @param hit_pointer
 * @return The position -- in the reference database -- of the subsequence
 * corresponding to the key
 */
int index__get_extern_next_hit(const IndexType *index, const int key, int* hit_pointer);

/**
 * The current hit in a reference index (as field external to the index structure)
 * @param index
 * @param key
 * @param hit_pointer
 * @return The position -- in the reference database -- of the subsequence
 * corresponding to the key
 */
int index__get_extern_current_hit (const IndexType *index, const int *hit_pointer);

/**
 * Destroy an indexed
 * @param index
 */
void index__destroy(IndexType* index);


#endif /* _INDEX_H_ */
