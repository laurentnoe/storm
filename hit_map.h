#ifndef _HIT_MAP_H_
#define _HIT_MAP_H_

#include "util.h"
#include "alignment.h"
#include "reads_data.h"
#include "reference_data.h"

#define UNMAPPED -1

/**
 * @struct HitInfoType
 * Keep trace of one read hit (score, indel positions, etc)
 * @see HitMapType
 */

typedef struct HitInfoType{
  /** the score of the alignment */
  ScoreType score;
  /** a score alignment adjusted according to the multiple alignment with the reads already mapped in that zone */
  ScoreType adjusted_score;
  /** the nreference sequence number */
  int ref_id;
  /** the position in the reference where the read is mapped */
  int ref_start;
  int ref_end;
  /** the position in the read where the read is mapped */
  short read_start;
  short read_end;
  /** the alignment sense (+1/-1) */
  int sense;
  /** alignment traceback (2 symbols per Byte)*/
  unsigned char* traceback;
  /** alignment traceback symbol length */
  int traceback_seq_len;
} HitInfoType;



/**
 * @struct HitMapType
 * Keep trace of all the hits for all the reads
 * @see HitInfoType
 */

typedef struct HitMapType{
  /** number of reads */
  int size;
  /** max number of indels */
  int indel_count;
  /** number of mapped reads (stats) */
  int mapped;
  /** 2D array
   * (of size equal to (nb reads) x (fixed nb maxhits per read = MAP_DETAIL_SIZE ))
   *  of HitInfoType struct
   */
  HitInfoType** map;
} HitMapType;

/**
 * Creates a map.
 * @param size The number of reads
 * @param indel_count The maximum number of allowed indels per read
 * @return The address of the created map
 */
HitMapType* hit_map__create(int size, int indel_count);

/**
 * Initializes a hit map
 * @param map The map to initialize
 * @param size The number of reads
 * @param indel_count The maximum number of allowed indels per read
 */
void hit_map__init(HitMapType* map, int size, int indel_count);

/**
 * Updates a map with a new alignment
 * @param map The map to update
 * @param read_id The aligned read's index in the reads database
 * @param ref_id The reference id that hits the read
 * @param ref_pos The position where the aligned part of the reference starts
 * @param alignment The obtained alignment
 * @return The rank of the alignment if its score was good enough to include in the map
 * (between 0 and MAP_DETAIL_SIZE exclusive), or -1 otherwise.
 */
int hit_map__update(HitMapType* map, const int read_id, const int ref_id, const int ref_pos, const AlignmentType* alignment, int alignment_sense);

/**
 * Obtain the score for a read, when mapped on a given position
 * @param map
 * @param read_id The read index
 * @param ref_id The reference id of interest
 * @param ref_pos The position of interest in the reference genome
 * @param rank The rank of that score will be placed at this address
 * @return The score
 */
inline int hit_map__score_for(const HitMapType* map, const int read_id, const int ref_id, const int ref_pos);

/**
 * Quick display of the successfully matched reads: index, position on the reference, score, alignment length, code sequence, alignment summary
 * @param map
 */
void hit_map__display(const HitMapType* map);

/**
 * Output the hit_map data as a full SAM output (avoid the "GenomeMap" that keep the "best read" per "genome pos" ; here display the score_rank_max best matching per read)
 * @param map
 * @param reads_db
 * @param ref_dbs
 * @param sam_output
 * @param score_rank_max is the maximal number of position mapped per read (cannot be more than MAP_DETAIL_SIZE)
 */
void hit_map__generate_SAM_output(const HitMapType* map,
                                  const ReadsDBType* reads_db,
                                  const ReferenceDBType* ref_dbs, const int ref_dbs_size,
                                  FILE* sam_output, const int score_rank_max);
/**
 * Output the hit_map "Unmapped" reads as a FASTQ output
 * @param map
 * @param reads_db
 * @param fastq_output
 */
void hit_map__generate_unmapped_FASTQ_output(const HitMapType* map,
                                             const ReadsDBType* reads_db,
                                             FILE* fastq_output);

/**
 * Cleanup
 * @param map
 */
void hit_map__destroy(HitMapType* map);

#endif /* _HIT_MAP_H_ */
