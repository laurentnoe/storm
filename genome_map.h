#ifndef _GENOME_MAP_H_
#define _GENOME_MAP_H_

#define MAPPING_ALGO_MA 0
#define MAPPING_ALGO_GREEDY 1
#define MAPPING_ALGO_SWITCHER MAPPING_ALGO_MA

int map_greedy;

#include "hit_map.h"
#include "alignment.h"

#define ALIGNED_POSITION_IDX 0
#define INSERTED_POSITION_IDX 1
#define CONSENSUS_POSITION_TYPES 2

#define COMPRESSED_COUNTERS

/**
 * @struct GenomeMapReadAtPositionType
 * gives Read index (on the readdb/hitmap)
 * for one read on a given position of the genome
 */
typedef struct GenomeMapReadAtPositionType {
  /** index of the read mapped here */
  int read_idx;
  /** where (in the hitmap) is the data for this mapping position */
  int score_rank;
} GenomeMapReadAtPositionType;


/**
 * @struct GenomeMapPositionType
 * gives Mapping Information (reads mapped and statistics)
 * for one position of the genome
 */
typedef struct GenomeMapPositionType {
  /** index of the reads mapped here and their hitmap pos*/
  GenomeMapReadAtPositionType * read_at_pos;
  /** index of the reads mapped here and their hitmap pos*/
  int read_at_pos_size;

#ifdef COMPRESSED_COUNTERS
  /** frequencies of each possible symbol (including gap) in mapped reads overlapping this position */
  unsigned aligned_positions_freqs;
  /** frequencies of each possible symbol (including NULL) in mapped reads inserted at this position */
  unsigned* inserted_positions_freqs;
#else
  /** frequencies of each possible symbol (including gap) in mapped reads overlapping this position */
  int *aligned_positions_freqs;
  /** frequencies of each possible symbol (including NULL) in mapped reads inserted at this position */
  int **inserted_positions_freqs;
#endif
} GenomeMapPositionType;


/**
 * @struct GenomeMapType
 *  set of structures needed for storing data
 *  and during mapping
 */

typedef struct GenomeMapType {
  /** 2D array (of size equal to ref number x ref size) of pointers
   *  to a GenomeMapPositionType struct
   *  (set to NULL when no read is mapped)
   *  @see GenomeMapPositionType
   */
  GenomeMapPositionType*** g_maps;
  /** Hitmap used to remap reads (of reads_db) on the references (ref_dbs) @see HitMapType
   */
  HitMapType* hitmap;
  /** Structures and number of References being used @see ReferenceDBType
   */
  /** @{ */
  ReferenceDBType* ref_dbs;
  int ref_dbs_size;
  /** @} */
  /** Reads database
   * @see ReadsDBType
   */
  ReadsDBType* reads_db;
  /** Indices of the reads (in reads_db), sorted according to the score
   */
  int* sorted_read_indices;
} GenomeMapType;

/**
 * Creates an empty genome map
 * @param hitmap The hitmap that will help populate the genome map
 * @param ref_dbs The reference sequences
 * @param ref_dbs_size The reference number of sequences
 * @param reads_dd The reads database
 * @return The genome_map to create
 */
GenomeMapType* genome_map__create(HitMapType* hitmap, ReferenceDBType* ref_dbs, int ref_dbs_size, ReadsDBType* reads_db);

/**
 * Build the map of the genomes
 * @param genome_map The genome map to build
 */
void genome_map__build(GenomeMapType* genome_map, const ScoreType match, const ScoreType mismatch);

/**
 * Outputs the mapped reads in the SAM fomat (http://samtools.sourceforge.net)
 * @param genome_map
 * @param sam_output
 */
void genome_map__generate_SAM_output(GenomeMapType* genome_map, FILE* sam_output);

/**
 * Quick display of pairs ref_pos: read_index for each position with a corresponding mapped read
 * @param genome_map
 */
void genome_map__display(const GenomeMapType* genome_map);

/**
 * Free the memory occupied by the genome map
 * @param genome_map
 */
void genome_map__destroy(GenomeMapType* genome_map);

#endif /* _GENOME_MAP_H_ */
