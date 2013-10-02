#ifndef _GENOME_MAP_H_
#define _GENOME_MAP_H_

#define MAPPING_ALGO_MA 0
#define MAPPING_ALGO_GREEDY 1
#define MAPPING_ALGO_SWITCHER MAPPING_ALGO_MA

int map_greedy;

#include "hit_map.h"

#define ALIGNED_POSITION_IDX 0
#define INSERTED_POSITION_IDX 1
#define CONSENSUS_POSITION_TYPES 2

#define COMPRESSED_COUNTERS

/**
 * @struct GenomeMapReadAtPosition
 *  gives Read index (on the readdb/hitmap)
 *  for one read on a given position of the genome
 */
typedef struct GenomeMapReadAtPositionType {
  /** index of the read mapped here */
  int read_idx;
  /** where (in the hitmap) is the data for this mapping position */
  int score_rank;
} GenomeMapReadAtPositionType;


/**
 * @struct GenomeMapPositionType
 *  gives Mapping Information (reads mapped and statistics)
 *  for one position of the genome
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
   */
  GenomeMapPositionType*** g_maps;
  /** Hitmap used to remap reads (of reads_db) on the references (ref_dbs)
   */
  HitMapType* hitmap;
  /** References being used
   */
  ReferenceDBType* ref_dbs;
  int ref_dbs_size;
  /** Reads database
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
void genome_map__build(GenomeMapType* genome_map);

/**
 * Outputs the mapped reads in the SAM fomat (http://samtools.sourceforge.net)
 * @param map
 * @param ref_db
 * @param reads_db
 * @param sam_output
 */
void genome_map__generate_SAM_output(GenomeMapType* genome_map, FILE* sam_output);

/**
 * Quick display of pairs ref_pos: read_index for each position with a corresponding mapped read
 */
void genome_map__display(const GenomeMapType* genome_map);

/**
 * Free the memory occupied by the genome map
 */
void genome_map__destroy(GenomeMapType* genome_map);

void genome_map__radix_sort_reads(GenomeMapType* genome_map);
/**
 * Find the colors that appears the most often in the reads at the certain position.
 * The result is an integer, holding the IDs of each color, in decreasing order of their
 * frequency of appearance on that position, encoded on 4 bits starting with the least
 * significant. Only the colors which appear a significant number of times are considered.
 * (the condition is to appear at least SIGNIFICANT_PERCENTAGE % wrt the previous color).
 * The 4 bits allocated to each color should be interpreted as follows:
 * FCCC, where F is a flag, set to 1 if those 4 bits correspond to a color,
 * and CCC is the 3 bit representation of the 5 possible values, from 0 to 4.
 */
/*
int genome_map__get_direct_consensus_codes(int* counts, int reference_code);
*/

#endif /* _GENOME_MAP_H_ */
