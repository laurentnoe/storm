#ifndef _READS_DATA_H_
#define _READS_DATA_H_

#include "load_data.h"

#include <limits.h>
#include <stdlib.h>

#define MAX_HEADER_LEN   64

#define TITLE_LINE_MARKER "# Title: "
#define TITLE_LINE_MARKER_LEN 9

/**
 * @struct ReadDataType
 * data stored for one read
 * @see ReadDBType
 */

typedef struct ReadDataType {
  /** header descriptor in the "csfasta/fastq" file */
  char      *info;
#ifndef NUCLEOTIDES
  /** first color (synch color, not used in the alignment process) */
  CODE_TYPE  first_base;
  /** first quality (associated to synch color) */
  QUAL_TYPE  first_qual;
#endif
  /** color space read quality (compressed in 2bits per qual) */
  QUAL_TYPE *quality;
  /** color space read sequence (compressed in 2bits per code) */
  CODE_TYPE *sequence;
} ReadDataType;


/**
 * @struct ReadsDBType
 * data stored for a set of reads
 * @see ReadDataType
 */
typedef struct ReadsDBType {
  /** descriptor name of the set of reads */
  char* name;
  /** length of the reads (all must have the same !!) */
  int read_len;
  /** number of reads */
  long size;
  /** array storing the set of reads */
  ReadDataType* reads;
} ReadsDBType;


/**
 * Reverse-complement a read
 */
void read__reverse(const ReadDataType* src, ReadDataType* dest, int len);

/**
 * Sort a read database (alphanumerical order)
 */
int sort_reads_db(ReadsDBType * db);

/**
 * Loads a reads database
 */
int load_reads_db(const char* reads_filename, const char* qual_filename, ReadsDBType* db);

/**
 * Destroys a read database
 */
void clear_reads_db(ReadsDBType* db);

#endif /* _READS_DATA_H_ */
