#include <math.h>
#include "genome_map.h"
#include "alignment.h"

int UNIQUENESS = 0;
int UNIQUENESS_SCORE_THRESHOLD = 0;
extern int delta_m;

#ifdef NUCLEOTIDES
#define CODE_COUNT BASE_CODE_COUNT
#else
#define CODE_COUNT COLOR_CODE_COUNT
#endif

#define GENOME_MAP_SIZE         (genome_map->ref_dbs[ref_id].size)
#define GENOME_MAP_READS_COUNT  (genome_map->hitmap->size)
#define GENOME_MAP_READS_MAPPED (genome_map->hitmap->mapped)
#define GENOME_MAP_INDELS       (genome_map->hitmap->indel_count)

#define TRUSTED_COUNT_RELATIVE_DIFFERENCE 0.6
#ifndef NUCLEOTIDES
#define FORWARD_WINDOW_SIZE 4
#endif

#define GENOME_MAP_NO_CODE CODE_COUNT


#define CODE_ORDER__EXISTING_CODE_FLAG 0x8
#define CODE_ORDER__EXISTING_CODE_MASK 0x7
#define CODE_ORDER__BITS_PER_CODE 4
#define CODE_ORDER__REAL_POS(p)        ((p) << 2)
#define CODE_ORDER__ADD(i, value, pos) {(i) |= (((value) | CODE_ORDER__EXISTING_CODE_FLAG) << CODE_ORDER__REAL_POS(pos));}
#define CODE_ORDER__POP(i, result)     {result = (i) & CODE_ORDER__EXISTING_CODE_MASK; (i) >>= CODE_ORDER__BITS_PER_CODE;}
#define CODE_ORDER__HAS_MORE(i)        ((i) & CODE_ORDER__EXISTING_CODE_FLAG)

#ifdef COMPRESSED_COUNTERS

#define COUNTER_TYPE unsigned

#define COUNTER_MASK 0x3F
#define COUNTER_MASK_SIZE 6

#define COUNTER_MASK_SHIFT(code)               ((code) * COUNTER_MASK_SIZE)
#define COUNTER_MASK_FOR(code)          ((COUNTER_TYPE)COUNTER_MASK << COUNTER_MASK_SHIFT(code))
#define COUNTER_GET(counter, code)        (((COUNTER_TYPE)(counter) >> COUNTER_MASK_SHIFT(code)) & COUNTER_MASK)
#define COUNTER_SET(counter, code, value)  {COUNTER_TYPE v = (counter) ; v = (v  & (~ (COUNTER_TYPE)COUNTER_MASK_FOR(code))) | (((COUNTER_TYPE)(value) & COUNTER_MASK) << COUNTER_MASK_SHIFT(code)); counter = v;}
#define COUNTER_INC(counter, code)         {COUNTER_TYPE u = COUNTER_GET(counter, code) + 1; COUNTER_SET(counter, code, u);}

#else

#define COUNTER_TYPE int*

#define COUNTER_GET(counter, code)          (counter)[code]
#define COUNTER_SET(counter, code, value) { (counter)[code] = (value) ; }
#define COUNTER_INC(counter, code)        { (counter)[code]++; }

#endif

extern char** cmd_line;
extern int    cmd_line_len;

int map_greedy = 0;

/**
 * Create the data structure for the needed (mapped) positions
 * @param genome_map The map to be updated
 * @param ref_id
 * @param re_pos position ref position
 */
static void genome_map__create_position_data(const GenomeMapType* genome_map, const int ref_id, const int ref_pos) {
  /* already allocated */
  if (genome_map->g_maps[ref_id][ref_pos]) {
    return;
  }
  /* otherwise create it and fill it */
  SAFE_FAILURE__ALLOC(genome_map->g_maps[ref_id][ref_pos], 1, GenomeMapPositionType);
  genome_map->g_maps[ref_id][ref_pos]->read_at_pos = NULL;
  genome_map->g_maps[ref_id][ref_pos]->read_at_pos_size = 0;
  genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs = NULL;

#ifndef COMPRESSED_COUNTERS
  genome_map->g_maps[ref_id][ref_pos]->aligned_positions_freqs  = NULL;
#endif

  if (!map_greedy) {
#ifdef COMPRESSED_COUNTERS
    genome_map->g_maps[ref_id][ref_pos]->aligned_positions_freqs  = 0;
    if (GENOME_MAP_INDELS) {
      SAFE_FAILURE__ALLOC(genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs, GENOME_MAP_INDELS, unsigned int);
      int k;
      for (k = 0; k < GENOME_MAP_INDELS; ++k) {
         genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs[k] = 0;
      }
    }
#else
    SAFE_FAILURE__ALLOC(genome_map->g_maps[ref_id][ref_pos]->aligned_positions_freqs, CODE_COUNT + 1, int)
    if (GENOME_MAP_INDELS) {
      SAFE_FAILURE__ALLOC(genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs, GENOME_MAP_INDELS, int*);
      int k;
      for (k = 0; k < GENOME_MAP_INDELS; ++k) {
        SAFE_FAILURE__ALLOC(genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs[k], CODE_COUNT + 1, int);
      }
    }
    {
      int j;
      for (j = 0; j <= CODE_COUNT ; ++j) {
        genome_map->g_maps[ref_id][ref_pos]->aligned_positions_freqs[j] = 0;
      }
    }
    genome_map->g_maps[ref_id][ref_pos]->aligned_positions_freqs[NTH_CODE(genome_map->ref_dbs[ref_id].sequence, ref_pos)] = 1;
    if (genome_map->hitmap && genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs) {
      int k;
      for (k = 0; k < GENOME_MAP_INDELS; ++k) {
        int j;
        for (j = 0; j <= CODE_COUNT ; ++j) {
          genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs[k][j] = 0;
        }
      }
    }
#endif
  }
}

/**
 * Access to map positions: set
 */
static void genome_map__set_read_idx_at(const GenomeMapType* genome_map, const int ref_id, const int ref_pos, const int read_id, const int score_rank) {
  int rank = 0;
  /* only creates the data if null. This ensures counts != null */
  genome_map__create_position_data(genome_map, ref_id, ref_pos);
  if (genome_map->g_maps[ref_id][ref_pos]->read_at_pos == NULL) {
    genome_map->g_maps[ref_id][ref_pos]->read_at_pos_size = 1;
    SAFE_FAILURE__ALLOC(genome_map->g_maps[ref_id][ref_pos]->read_at_pos,genome_map->g_maps[ref_id][ref_pos]->read_at_pos_size,GenomeMapReadAtPositionType);
  } else {
    rank = genome_map->g_maps[ref_id][ref_pos]->read_at_pos_size;
    genome_map->g_maps[ref_id][ref_pos]->read_at_pos_size++;
    SAFE_FAILURE__REALLOC(genome_map->g_maps[ref_id][ref_pos]->read_at_pos,genome_map->g_maps[ref_id][ref_pos]->read_at_pos_size,GenomeMapReadAtPositionType);
  }
  genome_map->g_maps[ref_id][ref_pos]->read_at_pos[rank].read_idx   = read_id;
  genome_map->g_maps[ref_id][ref_pos]->read_at_pos[rank].score_rank = score_rank;
}

/**
 * Creates an empty genome map
 * @param hitmap The hitmap that will help populate the genome map
 * @param ref_dbs The reference sequences
 * @param ref_dbs_size The reference number of sequences
 * @param reads_dd The reads database
 * @return The genome_map to create
 */
GenomeMapType* genome_map__create(HitMapType* hitmap, ReferenceDBType* ref_dbs, int ref_dbs_size, ReadsDBType* reads_db) {
  GenomeMapType* genome_map;
  SAFE_FAILURE__ALLOC(genome_map, 1, GenomeMapType);
  genome_map->hitmap        = hitmap;
  genome_map->ref_dbs       = ref_dbs;
  genome_map->ref_dbs_size  = ref_dbs_size;
  genome_map->reads_db      = reads_db;
  SAFE_FAILURE__ALLOC(genome_map->g_maps, ref_dbs_size, GenomeMapPositionType**);
  int ref_id;
  for(ref_id = 0; ref_id < ref_dbs_size; ++ref_id) {
    SAFE_FAILURE__ALLOC(genome_map->g_maps[ref_id], GENOME_MAP_SIZE, GenomeMapPositionType*);
    memset(genome_map->g_maps[ref_id],0,(GENOME_MAP_SIZE)*sizeof(GenomeMapPositionType*));
  }

  /* by default dont sort the read by their score */
  SAFE_FAILURE__ALLOC(genome_map->sorted_read_indices, GENOME_MAP_READS_MAPPED, int);
  int i_mapped;
  int j_index = 0;
  for (i_mapped = 0; i_mapped < GENOME_MAP_READS_COUNT; ++i_mapped)
    if (genome_map->hitmap->map[i_mapped])
      genome_map->sorted_read_indices[j_index++] = i_mapped;
  return genome_map;
}


/* ********************************************************************************
 *  Mapping version 1: greedy
 */

/**
 * Attempt to map a read
 * @param genome_map The map be used and updated if the read is mapped
 * @param ref_id The reference genome identifier
 * @param ref_pos The reference genome position
 * @return The position where the read is mapped with best score
 */
static int genome_map__map_read_greedy(const GenomeMapType* genome_map, const int read_id) {
  if (genome_map->hitmap->map[read_id]) {
    if ((genome_map->hitmap->map[read_id][0].ref_start) != UNMAPPED) {
      int ref_id  = genome_map->hitmap->map[read_id][0].ref_id;
      int ref_pos = genome_map->hitmap->map[read_id][0].ref_start;
      if ((UNIQUENESS == 0) || (genome_map->hitmap->map[read_id][0].score - genome_map->hitmap->map[read_id][1].score) >= UNIQUENESS_SCORE_THRESHOLD) {
        genome_map__set_read_idx_at(genome_map, ref_id, ref_pos, read_id, 0);
        return ref_pos;
      }
    }
  }
  return UNMAPPED;
}


/**
 * Build a genome map for a database of mapped reads - greedy approach
 * @param genome_map The map to be updated
 */
static void genome_map__build_greedy(GenomeMapType* genome_map) {
  /* rought read mapping (no score sort here, since no contextual multiple alignment) */
  int read_id;
  for (read_id = 0; read_id < genome_map->hitmap->size; ++read_id) {
    genome_map__map_read_greedy(genome_map, read_id);
  }

  /* compute and print coverage when asked */
  if (VERBOSITY >= VERBOSITY_MODERATE) {
    int ref_id;
    for (ref_id = 0; ref_id < genome_map->ref_dbs_size; ++ref_id) {
      int ref_coverage = 0;
      int reads_coverage = 0;
      int ref_pos;
      for (ref_pos = 0; ref_pos < GENOME_MAP_SIZE; ++ref_pos)
        if (genome_map->g_maps[ref_id][ref_pos] != NULL)  {
          ref_coverage  ++;
          reads_coverage += genome_map->g_maps[ref_id][ref_pos]->read_at_pos_size;
        }
      printf("Reference %d coverage: %5.2f%%. Actually mapped reads: %5.2f%% (%d).\n", ref_id+1, ref_coverage * 100.0 / genome_map->ref_dbs[ref_id].size, reads_coverage * 100.0 / genome_map->hitmap->size, reads_coverage);
    }
  }
}

/* ********************************************************************************
 *  Mapping version 2: read multiple alignment based
 */
static inline ScoreType genome_map__get_sorted_score(GenomeMapType* genome_map, int j_index) {
  return genome_map->hitmap->map[genome_map->sorted_read_indices[j_index]][0].score;
}

static inline HitInfoType* genome_map__get_sorted_hit(GenomeMapType* genome_map, int j_index) {
  return genome_map->hitmap->map[genome_map->sorted_read_indices[j_index]];
}

inline void genome_map__radix_sort_reads(GenomeMapType* genome_map) {
  int i = 0;
#define BITS_IN_DIGIT 4
#define BUCKETS (1 << (BITS_IN_DIGIT))
#define DIGITS  ((sizeof(ScoreType) * 8) / BITS_IN_DIGIT)
#define BUCKET_MASK ((BUCKETS) - 1)
#define GET_BUCKET(i, j_index) (BUCKET_MASK - ((genome_map__get_sorted_score(genome_map, (j_index)) >> ((i) * BITS_IN_DIGIT) ) & BUCKET_MASK))

  int count[BUCKETS];
  int index[BUCKETS];
  int *tmp;
  SAFE_FAILURE__ALLOC(tmp, GENOME_MAP_READS_MAPPED, int);

  for (i = 0 ; i < DIGITS ; i++) {
    /* Count how many entries are in each bucket */
    memset (count, 0, sizeof (count));
    {
      int j_index;
      for (j_index = 0; j_index < GENOME_MAP_READS_MAPPED; ++j_index)
        count[GET_BUCKET(i, j_index)]++;
    }
    /* Determine the position of each bucket in the destination */
    index[0] = 0;
    {
      int j;
      for (j = 1; j < BUCKETS; ++j) {
        index[j] = index[j-1] + count[j-1];
      }
    }

    /* Split into buckets */
    {
      int j_index;
      for (j_index = 0; j_index < GENOME_MAP_READS_MAPPED; ++j_index) {
        tmp[index[GET_BUCKET(i, j_index)]++] =
          genome_map->sorted_read_indices[j_index];
      }
    }

    /* Swap the two index vectors */
    {
      int  * swap = tmp;
      tmp = genome_map->sorted_read_indices;
      genome_map->sorted_read_indices = swap;
    }
  }
  free(tmp);
}

/**
 * Find the code that appears the most often in the reads at the certain position.
 * If there is a tie, and the reference code is one of the most frequent codes,
 * then the reference code is returned.
 * Otherwise, the first code with the highest number of appearances is returned.
 */
inline int genome_map__get_direct_consensus_code(const GenomeMapType* genome_map, const int ref_id, const int ref_pos, const int offset, const int reference_code) {

  /* no read mapped : set to the "reference" code */
  if (genome_map->g_maps[ref_id][ref_pos] == NULL) {
    return reference_code;
  }

  /* get the best code (inserted at 0, 1 ... /or/ exactly matched) */
  COUNTER_TYPE counts = (offset == -1) ? genome_map->g_maps[ref_id][ref_pos]->aligned_positions_freqs :  genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs[offset];

  int best_code = reference_code;
  int i;
  for (i = 0; i <= CODE_COUNT; ++i) {
    if (COUNTER_GET(counts, i) > COUNTER_GET(counts, best_code)) {
      best_code = i;
    }
  }

  if (COUNTER_GET(counts, best_code) == 0) {
    best_code = GENOME_MAP_NO_CODE;
  }

  return best_code;
}


#ifndef NUCLEOTIDES

/**
 * Find the color that appears the most often in the reads at the certain position.
 * If there is a tie, and the reference color is one of the most frequent colors,
 * then the reference color is returned.
 * Otherwise, the first color with the highest number of appearances is returned.
 * @param genome_map The map being used to get the direct consensus
 * @param ref_id The reference genome identifier
 * @param ref_pos The reference genome position
 * @param offset is -1 (no reference insertion) or a value lower than the maximal indel counts
 * @param reference_color
 * @return
 */

CODE_TYPE crt_checksum = CODE_IDENTITY;

inline int genome_map__get_consensus_color(const GenomeMapType* genome_map, const int ref_id, const int ref_pos, const int offset, const int reference_color) {

  int         best_color = reference_color;
  CODE_TYPE     checksum = crt_checksum;

  int window = 0;
  int fwd_i = ref_pos, fwd_j = offset;

  best_color = genome_map__get_direct_consensus_code(genome_map, ref_id, ref_pos, offset, reference_color);

  if (best_color != reference_color) {

    if (fwd_j == -1) {
      checksum = COMPOSE_SAFE(
                              checksum,
                              COMPOSE_SAFE(
                                           reference_color,
                                           best_color));
    } else {
      checksum = COMPOSE_SAFE(
                              checksum,
                              best_color);
    }

    while (window < FORWARD_WINDOW_SIZE) {
      ++fwd_j;
      if (fwd_j >= genome_map->hitmap->indel_count ||
          genome_map__get_direct_consensus_code(genome_map, ref_id, fwd_i, fwd_j, GENOME_MAP_NO_CODE) == GENOME_MAP_NO_CODE) {
        fwd_j = -1;
        ++fwd_i;
      }
      if (fwd_i >= genome_map->ref_dbs[ref_id].size) {
        break;
      }

      if (fwd_j == -1) {
        checksum = COMPOSE_SAFE(
                                checksum,
                                COMPOSE_SAFE(
                                             NTH_CODE(genome_map->ref_dbs[ref_id].sequence, fwd_i),
                                             genome_map__get_direct_consensus_code(
                                                                                   genome_map, ref_id, fwd_i, -1, NTH_CODE(genome_map->ref_dbs[ref_id].sequence, fwd_i))));
      } else {
        checksum = COMPOSE_SAFE(
                                checksum,
                                genome_map__get_direct_consensus_code(genome_map, ref_id, fwd_i, fwd_j, GENOME_MAP_NO_CODE));
      }
      if(checksum == CODE_IDENTITY) {
        break;
      }
      ++window;
    }
    if (checksum != CODE_IDENTITY) {
      if (offset == -1) {
        best_color = reference_color;
      } else {
        best_color = GENOME_MAP_NO_CODE;
      }
    }
  }
  if (offset >= 0) {
    crt_checksum = COMPOSE_SAFE(crt_checksum, best_color);
  } else {
    crt_checksum = COMPOSE_SAFE(crt_checksum, COMPOSE_SAFE(
                                                           reference_color, best_color
                                                           ));
  }
  return best_color;
}

#endif


/**
 * Update code frequency in stacked reads
 * @param genome_map The genome map being used to update counters
 * @param ref_id The reference genome identifier
 * @param ref_pos The reference genome position
 * @param offset is -1 if no insertion is considered, otherwise >= 0
 * @param code The code counter being incremented
 */

inline void genome_map__update_consensus_code(GenomeMapType* genome_map, int ref_id, int ref_pos, int offset, int code) {
  /* only creates the data if null. This ensures counts != null */
  genome_map__create_position_data(genome_map, ref_id, ref_pos);
  /* if offset >= 0, this means that its a given gap position (0, 1 ...) on the reference, otherwise offset == -1 */
  if (offset >= 0) {
    COUNTER_INC(genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs[offset], code);
  } else {
    COUNTER_INC(genome_map->g_maps[ref_id][ref_pos]->aligned_positions_freqs, code);
  }
}

/**
 * Recompute the alignment score of a read, according to other reads it overlaps in the mapping
 * @param genome_map The genome map being used to adjust scores
 * @param read_id The read_id to get access to the hitmap stats
 * @param rank The rank in the hitmap
 * @return the score adjusted according to the previous reads mapped in the genome map
 */
ScoreType genome_map__compute_adjusted_score(GenomeMapType* genome_map, int read_id, int rank) {
  ScoreType      score = 0;
  int            indel = 0;
  int            ref_id        = genome_map->hitmap->map[read_id][rank].ref_id;
  int            ref_pos       = genome_map->hitmap->map[read_id][rank].ref_start;
  unsigned char* traceback     = genome_map->hitmap->map[read_id][rank].traceback;
  int            traceback_len = genome_map->hitmap->map[read_id][rank].traceback_seq_len;

  if (ref_pos == UNMAPPED)
    return MIN_SCORE;

  int i;
  for (i = 0; i < traceback_len; ++i) {
    unsigned char traceback_cell      = TRACEBACK_SEQ_GET(traceback, i);
    unsigned char traceback_cell_type = GET_PAIR_TYPE(traceback_cell);

    switch (traceback_cell_type) {

    case TRACEBACK_PAIR_MATCH :
      {
        CODE_TYPE ref_code = NTH_CODE(genome_map->ref_dbs[ref_id].sequence, ref_pos);
        score += (ref_code ==
                  genome_map__get_direct_consensus_code(genome_map, ref_id, ref_pos, -1, ref_code)) ? MATCH : MISMATCH;
        ++ref_pos;
        indel = 0;
      }
      break;

    case TRACEBACK_PAIR_READ_DELETION :
      score += (genome_map__get_direct_consensus_code(genome_map, ref_id, ref_pos, -1, GENOME_MAP_NO_CODE)
                == GENOME_MAP_NO_CODE) ? MATCH : MISMATCH;
      ++ref_pos;
      indel = 0;
      break;

    case TRACEBACK_PAIR_CODE_MISMATCH :
#ifndef NUCLEOTIDES
    case TRACEBACK_PAIR_COLORBASE_MISMATCH :
#endif
      {
        CODE_TYPE read_code = (traceback_cell & TRACEBACK_CODE_MASK);
        score += (read_code ==
                  genome_map__get_direct_consensus_code(genome_map, ref_id, ref_pos, -1, read_code)) ? MATCH : MISMATCH;
        ++ref_pos;
        indel = 0;
      }
      break;

    case TRACEBACK_PAIR_READ_INSERTION :
      {
        CODE_TYPE read_code = (traceback_cell & TRACEBACK_CODE_MASK);
        score += (read_code ==
                  genome_map__get_direct_consensus_code(genome_map, ref_id, ref_pos, indel, read_code)) ? MATCH : MISMATCH;
        ++indel;
      }
      break;
    }
  }
  return score;
}

/**
 * Build a genome map for a database of mapped reads: multiple alignment approach
 */
void genome_map__build_contextual(GenomeMapType* genome_map) {

  /* 1) sort reads by best score */
  genome_map__radix_sort_reads(genome_map);

  /* 2) for each read */
  int i;
  for (i = 0; i < GENOME_MAP_READS_MAPPED; ++i) {
    /* get the target read (best score first) */
    int read_id     = genome_map->sorted_read_indices[i];
    int best_rank   = UNMAPPED;
    ScoreType best_score = MIN_SCORE;

    /* 2.1) recompute the read scores wrt the "consensus" */
    {
      int rank;
      for (rank = 0; rank < MAP_DETAIL_SIZE && genome_map->hitmap->map[read_id][rank].ref_start != UNMAPPED; ++rank) {
        ScoreType score = genome_map->hitmap->map[read_id][rank].adjusted_score = genome_map__compute_adjusted_score(genome_map, read_id, rank);
        if (score > best_score) {
          best_rank  = rank;
          best_score = score;
        }
      }
    }

    /* 2.2) map the read in the best available place and update the consensus */
    if (best_rank != UNMAPPED && ((UNIQUENESS == 0) || (genome_map->hitmap->map[read_id][0].adjusted_score - genome_map->hitmap->map[read_id][1].adjusted_score) > UNIQUENESS_SCORE_THRESHOLD)) {
      /* map the read on the genome and update the genome consensus */
      int ref_id  = genome_map->hitmap->map[read_id][best_rank].ref_id;
      int ref_pos = genome_map->hitmap->map[read_id][best_rank].ref_start;
      genome_map__set_read_idx_at(genome_map, ref_id, ref_pos, read_id, best_rank);
      /* adjust consensus */
      int insertions = -1;
      int k;
      for (k = 0; k < genome_map->hitmap->map[read_id][best_rank].traceback_seq_len; ++k) {
        unsigned char traceback_cell      = TRACEBACK_SEQ_GET(genome_map->hitmap->map[read_id][best_rank].traceback, k);
        unsigned char traceback_cell_type = GET_PAIR_TYPE(traceback_cell);

        switch (traceback_cell_type) {

        case TRACEBACK_PAIR_READ_INSERTION :
          insertions++;
          genome_map__update_consensus_code(genome_map, ref_id, ref_pos, insertions, (traceback_cell & TRACEBACK_CODE_MASK) /* read_code */);
          break;

        case TRACEBACK_PAIR_MATCH :
          insertions = -1;
          genome_map__update_consensus_code(genome_map, ref_id, ref_pos, insertions, NTH_CODE(genome_map->ref_dbs[ref_id].sequence, ref_pos) /* ref_code */);
          break;

        case TRACEBACK_PAIR_CODE_MISMATCH :
          insertions = -1;
          genome_map__update_consensus_code(genome_map, ref_id, ref_pos, insertions, (traceback_cell & TRACEBACK_CODE_MASK) /* read_code */);
          break;

        case TRACEBACK_PAIR_READ_DELETION :
          insertions = -1;
          genome_map__update_consensus_code(genome_map, ref_id, ref_pos, insertions, GENOME_MAP_NO_CODE);
          break;

#ifndef NUCLEOTIDES
        case TRACEBACK_PAIR_COLORBASE_MISMATCH :
          insertions = -1;
          genome_map__update_consensus_code(genome_map, ref_id, ref_pos, insertions, (traceback_cell & TRACEBACK_CODE_MASK) /* read_code */);
          break;
#endif
        default:
          break;
        }

        /* update the "not insertion here" codes with NO_CODE for the read (means that there is no insertion at this ref_pos so ...) */
        int p;
        for (p = insertions + 1; p < GENOME_MAP_INDELS; ++p) {
          genome_map__update_consensus_code(genome_map, ref_id, ref_pos, p, GENOME_MAP_NO_CODE);
        }

        /* go to the next position on the genome (if not inside an insertion) */
        if (insertions < 0) {
          ref_pos++;
        }
      }
    }
  }

  /* compute and print coverage when asked */
  if (VERBOSITY >= VERBOSITY_MODERATE) {
    int ref_id;
    for (ref_id = 0; ref_id < genome_map->ref_dbs_size; ++ref_id) {
      int ref_coverage = 0;
      int reads_coverage = 0;
      int ref_pos;
      for (ref_pos = 0; ref_pos < GENOME_MAP_SIZE; ++ref_pos)
        if (genome_map->g_maps[ref_id][ref_pos] != NULL)  {
          ref_coverage  ++;
          reads_coverage += genome_map->g_maps[ref_id][ref_pos]->read_at_pos_size;
        }
      printf("Reference %d coverage: %5.2f%%. Actually mapped reads: %5.2f%% (%d).\n", ref_id+1, ref_coverage * 100.0 / genome_map->ref_dbs[ref_id].size, reads_coverage * 100.0 / genome_map->hitmap->size, reads_coverage);
    }
  }
}

void genome_map__build(GenomeMapType* genome_map) {
  if (map_greedy) {
    genome_map__build_greedy(genome_map);
  } else {
    genome_map__build_contextual(genome_map);
  }
}


/**
 * Outputs the mapped reads in the SAM fomat (http://samtools.sourceforge.net)
 * @param map
 * @param ref_db
 * @param reads_db
 * @param sam_output
 */
void genome_map__generate_SAM_output(GenomeMapType* genome_map, FILE* sam_output) {
  ReferenceDBType* ref_dbs = genome_map->ref_dbs;
  ReadsDBType* reads_db    = genome_map->reads_db;
  HitMapType* map          = genome_map->hitmap;

  /* Worst case: 1M1I1M1I ... => 4*read length [FIXME] : size of the matches runs (101M1IM1I) should be take into account + maximal number of indels should also be measured ... */
  int   CIGAR_max_size = reads_db->read_len << 2;
  char* CIGAR_STRING;

#ifdef NUCLEOTIDES
  int translated_read_max_size = reads_db->read_len + 1;
#else
  int translated_read_max_size = reads_db->read_len + 2;
#endif
  char* translated_read_string;

#define FLAG_EMPTY          0x0000
#define FLAG_PAIRED         0x0001 /* the read is paired in sequencing, no matter whether it is mapped in a pair */
#define FLAG_MAPPED_PAIRED  0x0002 /* the read is mapped in a proper pair (depends on the protocol, normally inferred during alignment) 1 */
#define FLAG_UNMAPPED       0x0004 /* the query sequence itself is unmapped */
#define FLAG_MATE_UNMAPPED  0x0008 /* the mate is unmapped */
#define FLAG_REVERSE        0x0010 /* strand of the query (0 for forward; 1 for reverse strand) */
#define FLAG_MATE_REVERSE   0x0020 /* strand of the mate */
#define FLAG_FIRST_MATE     0x0040 /* the read is the first read in a pair */
#define FLAG_SECOND_MATE    0x0080 /* the read is the second read in a pair */
#define FLAG_NOT_PRIMARY    0x0100 /* the alignment is not primary (a read having split hits may have multiple primary alignment records) */
#define FLAG_FAIL           0x0200 /* the read fails platform/vendor quality checks */
#define FLAG_DUPLICATE      0x0400 /* the read is either a PCR duplicate or an optical duplicate */
#define FLAG_SUPPLEMENTARY  0x0800 /* the read is a supplementary alignment (chimeric) */

  if (!sam_output) {
    ERROR__(("No file name provided for SAM output."));
    return;
  }

  /* Memory allocations */
  SAFE_FAILURE__ALLOC(CIGAR_STRING, CIGAR_max_size, char);
  SAFE_FAILURE__ALLOC(translated_read_string, translated_read_max_size, char);

  /* --- */

  /* Display file header */
  /*
  Type           | Tag    | Description
  ---------------+--------+------------------------------------------------------------------------
  HD - header    | VN*    | File format version.
                 | SO     | Sort order. Valid values are: unsorted, queryname or coordinate.
                 | GO     | Group order (full sorting is not imposed in a group). Valid values are:
                 |        |   none, query or reference.
  ---------------+--------+------------------------------------------------------------------------
  SQ - Sequence  | SN*    | Sequence name. Unique among all sequence records in the ﬁle. The value
       dictionary|        | of this ﬁeld is used in alignment records.
                 | LN*    | Sequence length.
                 | AS     | Genome assembly identiﬁer. Refers to the reference genome assembly in
                 |        | an unambiguous form. Example: HG18.
                 | M5     | MD5 checksum of the sequence in the uppercase (gaps and space removed)
                 | UR     | URI of the sequence
                 | SP     | Species.
  ---------------+--------+------------------------------------------------------------------------
  RG - read group| ID*    | Unique read group identiﬁer. The value of the ID ﬁeld is used in the
                 |        | RG tags of alignment records.
                 | SM*    | Sample (use pool name where a pool is being sequenced)
                 | LB     | Library
                 | DS     | Description
                 | PU     | Platform unit (e.g. lane for Illumina or slide for SOLiD); should be a
                 |        | full, unambiguous identiﬁer
                 | PI     | Predicted median insert size (maybe different from the actual median
                 |        | insert size)
                 | CN     | Name of sequencing center producing the read.
                 | DT     | Date the run was produced (ISO 8601 date or date/time).
                 | PL     | Platform/technology used to produce the read.
  ---------------+--------+------------------------------------------------------------------------
  PG - Program   | ID*    | Program name
                 | VN     | Program version
                 | CL     | Command line
  ---------------+--------+------------------------------------------------------------------------
  CO - Comment   |        | One-line text comment
  */

#define DISPLAY_NAME(db) ((db).name ? (db).name : "*")

  /* SAM Version:1.4, Sorting Order:coordinate */
  fprintf(sam_output, "@HD\tVN:1.4\tSO:coordinate\n");

  /* Header section first */
  {
    int ref_id;
    for (ref_id = 0; ref_id < genome_map->ref_dbs_size; ++ref_id) {
#ifdef NUCLEOTIDES
      fprintf(sam_output, "@SQ\tSN:%s\tLN:%ld\n", DISPLAY_NAME(ref_dbs[ref_id]), (long)(ref_dbs[ref_id].size));
#else
      fprintf(sam_output, "@SQ\tSN:%s\tLN:%ld\n", DISPLAY_NAME(ref_dbs[ref_id]), (long)(ref_dbs[ref_id].size + 1));
#endif
    }
    fprintf(sam_output, "@RG\tID:%s\tSM:%s\n", DISPLAY_NAME(*reads_db),  DISPLAY_NAME(*reads_db));
    fprintf(sam_output, "@PG\tID:%s\tPN:%s\tCL:", PROGRAM_NAME, PROGRAM_NAME);
    int i;
    for (i = 0; i < cmd_line_len; ++i) {
      fprintf(sam_output, "%s ", cmd_line[i]);
    }
    fprintf(sam_output, "\tVN:%s\n",PROGRAM_VERSION);
  }

  /* Display read */
  /*
  Field | Regular expression    | Range         | Description
  ------+-----------------------+---------------+--------------------------------------------------
  QNAME | ([^ \t\n\r])+         |               | Query pair NAME if paired; Query NAME if unpaired
  ------+-----------------------+---------------+--------------------------------------------------
  FLAG  | [0-9]+                | [0,2^16-1]    | bitwise FLAG
  ------+-----------------------+---------------+--------------------------------------------------
  RNAME | ([^ \t\n\r@-])+       |               | Reference sequence name
  ------+-----------------------+---------------+--------------------------------------------------
  POS   | [0-9]+                | [0,2^29-1]    | 1-based leftmost POSition/coordinate of the
        |                       |               | clipped sequence
  ------+-----------------------+---------------+--------------------------------------------------
  MAPQ  | [0-9]+                | [0,2^8-1]     | MAPping Quality (phred-scaled posterior probability
        |                       |               |  that the mapping position of this read is incorrect)
  ------+-----------------------+---------------+--------------------------------------------------
  CIGAR | ([0-9]+[MIDNSHP])+|\* |               | extended CIGAR string
  ------+-----------------------+---------------+--------------------------------------------------
  MRNM  | ([^ \t\n\r@])+        |               | Mate Reference sequence NaMe; “=” if the same as <RNAME>
  ------+-----------------------+---------------+--------------------------------------------------
  MPOS  | [0-9]+                | [0,2^29-1]    | 1-based leftmost Mate POSition of the clipped sequence
  ------+-----------------------+---------------+--------------------------------------------------
  ISIZE | -?[0-9]+              | [-2^29, 2^29] | inferred Insert SIZE
  ------+-----------------------+---------------+--------------------------------------------------
  SEQ   | [acgtnACGTN.=]+|\*    |               | query SEQuence; “=” for a match to the reference;
        |                       |               | n/N/. for ambiguity; cases are not maintained
  ------+-----------------------+---------------+--------------------------------------------------
  QUAL  | [!-~]+|\*             | [0,93]        | query QUALity; ASCII-33 gives the Phred base quality
  ------+-----------------------+---------------+--------------------------------------------------
  TAG   | [A-Z][A-Z0-9]         |               | TAG
  ------+-----------------------+---------------+--------------------------------------------------
  VTYPE | [AifzH]               |               | Value TYPE
  ------+-----------------------+---------------+--------------------------------------------------
  VALUE | [^\t\n\r]             |               | match <VTYPE> (space allowed)
  */

  /*
   * 2_66_251_F3     16      merged  2241562 255     35M     *       0       0
   * CTACATTGCAGTTGAGTACGGCAGTCATCTAATAA     #%!!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     NM:i:1  CS:Z:T0330322312121303131221012131031032        CQ:Z:###################################
   */

  int ref_id;
  for (ref_id = 0; ref_id < genome_map->ref_dbs_size; ++ref_id) {
#ifndef NUCLEOTIDES
    CODE_TYPE ref_symbol = ref_dbs[ref_id].first_base;
#endif
    int ref_pos;
    for (ref_pos = 0; ref_pos < GENOME_MAP_SIZE; ++ref_pos) {
      if (genome_map->g_maps[ref_id][ref_pos]) {
        int k;
        for(k = 0; k < genome_map->g_maps[ref_id][ref_pos]->read_at_pos_size; ++k) {
          int read_id        = genome_map->g_maps[ref_id][ref_pos]->read_at_pos[k].read_idx;
          int score_rank     = genome_map->g_maps[ref_id][ref_pos]->read_at_pos[k].score_rank;
          ReadDataType* read = &(reads_db->reads[read_id]);
          int flag           = FLAG_EMPTY;
          if (map->map[read_id][score_rank].sense & ALIGNMENT_SENSE_REVERSE) {
            flag |= FLAG_REVERSE;
          }

          /* alignment quality for the first read pos mapped */
          int alignment_quality = 255;
          if (score_rank == 0) {
            int score1 = map->map[read_id][0].score, score2 = MAX(map->map[read_id][1].score,min_accepted_score);
            /* rough estimate */
            double n = (double)(score1 - score2)/(double)(delta_m);
            int mapq  = -4.343 * (n*log(.3));/* value between 0.25 and 0.33 */
            mapq = (int) (mapq - 4.343*log(.3));
            mapq = mapq >= 0  ? mapq : 0;
            mapq = mapq < 254 ? mapq : 254;
            alignment_quality = mapq;
          } else {
            int score1 = map->map[read_id][0].score, score2 = map->map[read_id][score_rank].score;
            /* rough estimate */
            double n = (double)(score2 - score1)/(double)(delta_m);
            int mapq  = -4.343 * (n*log(.3));/* value between 0.25 and 0.33 */
            mapq = (int) (mapq - 4.343*log(.3));
            mapq = mapq >= 0  ? mapq : 0;
            mapq = mapq < 254 ? mapq : 254;
            alignment_quality = mapq;
          }
          /* [FIXME] TODO: better computation , not just the score but -10 log_10 (Pr{mapping pos is wrong}) (or 255 if not available) */

          memset(CIGAR_STRING, 0x0, CIGAR_max_size);
          memset(translated_read_string, 0x0, translated_read_max_size);
          int edit_distance = alignment__traceback_to_CIGAR(map->map[read_id][score_rank].traceback, map->map[read_id][score_rank].traceback_seq_len,
                                                            ref_dbs[ref_id].sequence + COMPRESSED_IDX(ref_pos), COMPRESSED_OFFSET(ref_pos),
#ifndef NUCLEOTIDES
                                                            ref_symbol,
#endif
                                                            CIGAR_STRING, translated_read_string);

          /* Read name, flag, reference name, mapping position, alignment quality, CIGAR string */
          fprintf(sam_output, "%s\t%d\t%s\t%d\t%d\t%s\t", read->info, flag, DISPLAY_NAME(ref_dbs[ref_id]), (ref_pos+1) /* genome positions for 1 to N */, alignment_quality, CIGAR_STRING);
          /* The mate information (mate reference name, mate position, inserted size) is not supported yet (TODO) [FIXME] */
          fprintf(sam_output, "=\t0\t0\t");

          /* Sequence (translated) and quality */

          /* Reversing is not needed for sam... */
          /* if (flag & FLAG_REVERSE) {
           *   REVERSE_COMPLEMENT(translated_read_string);
           * }
           */

          /* display the read sequence (base format) */
          fprintf(sam_output, "%s\t", translated_read_string);

          /* display the quality (phread) */
          if (reads_db->reads[read_id].quality) {
            if (map->map[read_id][score_rank].sense & ALIGNMENT_SENSE_REVERSE) {
              int q;
              for (q = reads_db->read_len - 1; q >= 0; --q) {
                fprintf(sam_output, "%c", 33 + READ_QUALITY_LEVEL_UPPER_BOUNDS[NTH_QUAL(reads_db->reads[read_id].quality, q)]);
              }
#ifndef NUCLEOTIDES
              fprintf(sam_output, "%c", 33 + READ_QUALITY_LEVEL_UPPER_BOUNDS[reads_db->reads[read_id].first_qual]);
#endif
            } else {
              int q;
#ifndef NUCLEOTIDES
              fprintf(sam_output, "%c", 33 + READ_QUALITY_LEVEL_UPPER_BOUNDS[reads_db->reads[read_id].first_qual]);
#endif
              for (q = 0; q < reads_db->read_len; ++q) {
                fprintf(sam_output, "%c", 33 + READ_QUALITY_LEVEL_UPPER_BOUNDS[NTH_QUAL(reads_db->reads[read_id].quality, q)]);
              }
            }
          } else {
            /* quality replaced by a "*" if no quality */
            fprintf(sam_output, "*");
          }
          fprintf(sam_output, "\tAS:i:%d\tNM:i:%d\n", map->map[read_id][score_rank].score, edit_distance);
        }
      }
#ifndef NUCLEOTIDES
      ref_symbol = TRANSFORM(ref_symbol, NTH_CODE(ref_dbs[ref_id].sequence, ref_pos));
#endif
    }
  }
  fflush(sam_output);

  free(CIGAR_STRING);
  free(translated_read_string);
}

/**
 * Quick display of pairs ref_pos: read_index for each position with a corresponding mapped read
 * @param genome_map The genome_map to be printed
 */

void genome_map__display(const GenomeMapType* genome_map) {
  int ref_id;
  for (ref_id = 0; ref_id < genome_map->ref_dbs_size; ++ref_id) {
    int ref_pos;
    for (ref_pos = 0; ref_pos < GENOME_MAP_SIZE; ++ref_pos) {
      if (genome_map->g_maps[ref_id][ref_pos] != NULL) {
        int k;
        for (k = 0; k < genome_map->g_maps[ref_id][ref_pos]->read_at_pos_size; ++k) {
          int read_id    = genome_map->g_maps[ref_id][ref_pos]->read_at_pos[k].read_idx;
          int score_rank = genome_map->g_maps[ref_id][ref_pos]->read_at_pos[k].score_rank;
          int ref_pos_h  = genome_map->hitmap->map[read_id][score_rank].ref_start;
          int tb_len     = genome_map->hitmap->map[read_id][score_rank].traceback_seq_len;
          int score      = genome_map->hitmap->map[read_id][score_rank].score;
          int a_score    = genome_map->hitmap->map[read_id][score_rank].adjusted_score;
          printf("(%d,%d) : %d, %d (%d -- %d, %d, %d) \n", ref_id, ref_pos, read_id, score_rank, ref_pos_h, tb_len, score, a_score);
        }
      }
    }
  }
}

/**
 * Free the memory occupied by the genome map
 * @param genome_map The genome_map to be erased
 */
void genome_map__destroy(GenomeMapType* genome_map) {
  int ref_id;
  for (ref_id = 0; ref_id < genome_map->ref_dbs_size; ++ref_id) {
    int ref_pos;
    for (ref_pos = 0; ref_pos < GENOME_MAP_SIZE; ++ref_pos) {
      if (genome_map->g_maps[ref_id][ref_pos]) {

        /* free "read_at_pos" */
        if(genome_map->g_maps[ref_id][ref_pos]->read_at_pos) {
          free(genome_map->g_maps[ref_id][ref_pos]->read_at_pos);
          genome_map->g_maps[ref_id][ref_pos]->read_at_pos = NULL;
        }

        /* free "counters" and "insertion counters" */
#ifndef COMPRESSED_COUNTERS
        if (genome_map->g_maps[ref_id][ref_pos]->aligned_positions_freqs) {
          free(genome_map->g_maps[ref_id][ref_pos]->aligned_positions_freqs);
          genome_map->g_maps[ref_id][ref_pos]->aligned_positions_freqs = NULL;
        }
#endif
        if (genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs) {
#ifndef COMPRESSED_COUNTERS
          int j;
          for(j = 0; j < GENOME_MAP_INDELS; ++j) {
            free(genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs[j]);
            genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs[j] = NULL;
          }
#endif
          free(genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs);
          genome_map->g_maps[ref_id][ref_pos]->inserted_positions_freqs = NULL;
        }

        /* then free the structure */
        free(genome_map->g_maps[ref_id][ref_pos]);
        genome_map->g_maps[ref_id][ref_pos] = NULL;
      }
    }
    /* and free the pos tables for each reference  */
    free(genome_map->g_maps[ref_id]);
    genome_map->g_maps[ref_id] = NULL;
  }
  free(genome_map->g_maps);
  genome_map->g_maps = NULL;
  free(genome_map->sorted_read_indices);
  genome_map->sorted_read_indices = NULL;
}
