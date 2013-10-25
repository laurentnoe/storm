#ifndef _REFERENCE_DATA_H_
#define _REFERENCE_DATA_H_

#include "load_data.h"
#include "util.h"
/**
 * @struct Linked list of masked regions
 * on the ref sequence.
 */

typedef struct MaskedRegionType {
  /** first pos masked */
  int start;
  /** last pos masked */
  int end;
  /** next one */
  struct MaskedRegionType* next;
} MaskedRegionType;

/**
 * @struct Ref sequence descriptor
 */

typedef struct ReferenceDBType {
  /** fasta name */
  char* name;
  /** size (nucleotides or color) */
  int size;
#ifndef NUCLEOTIDES
  /** first base for color encoded sequence */
  CODE_TYPE first_base;
#endif

  /** "allocated" sequence pointer (larger than sequence to avoid SIMD checks)
   *   this one only must be freed
   */
  CODE_TYPE* sequence_alloc;
  /** "true" sequence pointer */
  CODE_TYPE* sequence;

  /** "allocated" sequence_masked pointer (larger than sequence_masked to avoid SIMD checks)
   *   this one only must be freed
   */
  CODE_TYPE* sequence_masked_alloc;
  /** "true" sequence of list of masked regions in flat "code" format
   */
  CODE_TYPE* sequence_masked;

  /** list of masked regions along the genome */
  MaskedRegionType* masked_regions;

} ReferenceDBType;

/**
 * Load reference sequence
 */
int load_reference_db(const char* reference_filename, int readlen, ReferenceDBType** db);

/**
 * Destroys a read database
 */
void clear_reference_db(ReferenceDBType* db);

#endif /* _REFERENCE_DATA_H_ */
