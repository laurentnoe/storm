#include <math.h>
#include "hit_map.h"

extern int UNIQUENESS;
extern int UNIQUENESS_SCORE_THRESHOLD;
extern int delta_m;

/**
 * Creates a map.
 * @param size
 * @return The address of the created map
 */
HitMapType* hit_map__create(int size, int indel_count) {
  /* for each read, memorize MAP_DETAIL_SIZE best scores */
  HitMapType* map;
  SAFE_FAILURE__ALLOC(map, 1, HitMapType);
  hit_map__init(map, size, indel_count);
  return map;
}

/**
 * Init a map structure.
 * @param map The map structure (already allocated)
 * @param size The map size (at the total number of reads)
 * @param indel_count The maximal number of indels (when aligning a read)
 */
void hit_map__init(HitMapType* map, int size, int indel_count) {
  map->size = size;
  map->indel_count = indel_count;
  map->mapped = 0;
  /* for each read, memorize MAP_DETAIL_SIZE best scores */
  SAFE_FAILURE__ALLOC(map->map, size, HitInfoType*);
  int read_id;
  for (read_id = 0; read_id < size; ++read_id) {
    map->map[read_id] = NULL;
  }
}

/**
 * Updates a map with a new alignment
 * @param map The map to update
 * @param read_id The aligned read's index in the reads database
 * @param ref_pos The position where the aligned part of the reference starts
 * @param alignment The obtained alignment
 * @return The rank of the alignment if its score was good enough to include in the map
 * (between 0 and MAP_DETAIL_SIZE exclusive), or UNMAPPED (-1) otherwise.
 */
int hit_map__update(HitMapType* map, const int read_id, const int ref_id, const int ref_pos, const AlignmentType* alignment, int alignment_sense) {


  /* (0) init the full table if this first time mapped */
  int mapped = 0;
  if (!(map->map[read_id])) {
    mapped = 1;
    SAFE_FAILURE__ALLOC(map->map[read_id], MAP_DETAIL_SIZE, HitInfoType);
    int k;
    for (k = 0; k < MAP_DETAIL_SIZE; ++k) {
      map->map[read_id][k].score             = MIN_SCORE;
      map->map[read_id][k].adjusted_score    = MIN_SCORE;
      map->map[read_id][k].ref_id            = UNMAPPED;
      map->map[read_id][k].ref_start         = UNMAPPED;
      map->map[read_id][k].ref_end           = UNMAPPED;
      map->map[read_id][k].read_start        = UNMAPPED;
      map->map[read_id][k].read_end          = UNMAPPED;
      map->map[read_id][k].sense             = 0;
      map->map[read_id][k].traceback         = NULL;
      map->map[read_id][k].traceback_seq_len = 0;
    }
  }

  HitInfoType* info = map->map[read_id];
  ScoreType score = alignment__get_score(alignment, alignment->best_i, alignment->best_j);

  /* (1) first search for possible duplicate in already mapped positions */
  int k;
  for (k = 0; k < MAP_DETAIL_SIZE && info[k].ref_id != UNMAPPED; ++k) {
    /* [FIXME] considered as duplicate if the "start" /or/ the "end" of the alignment is the same : well ... no so satisfying */
    if ((info[k].ref_id == ref_id) &&
      (
       ((info[k].ref_start == (ref_pos + alignment->best_j0)) && (info[k].read_start == alignment->best_i0))
       ||
       ((info[k].ref_end   == (ref_pos + alignment->best_j )) && (info[k].read_end   == alignment->best_i ))
       )) {
      if (score > info[k].score) {
      /* the new one is better : must update (replace) */
      break;
      } else {
      /* the old one was better or equivalent : do nothing and stop here (avoid duplicates) */
      return UNMAPPED;
      }
    }
  }

  /* (2) update if new or duplicate to erase ... */
  if (k == MAP_DETAIL_SIZE) k--; /* overflow if previous for loop ended
                                  * otherwise : ether "k" is the one to replace,
                                  *                or "k" is the first unmapped place.
                                  */

  int i = k;
  while (i >= 0 && score > info[i].score) i--; /* i is the first place than must not move
                                                * (possibly -1 if all must move)
                                                */

  /* score seached is smaller than everything before, and table already full, so no update needed */
  if (i == MAP_DETAIL_SIZE-1) {
    return UNMAPPED;
  }

  /* free the last one (pushed away) or free the replaced one (will be allocated again at the right "i"<"k") */
  if (info[k].traceback) {
    free(info[k].traceback);
    info[k].traceback = NULL;
  }

  /* shift to create one insert space at (i+1) */
  int j;
  for (j = k; j > i+1; --j) {
    info[j] = info[j-1];
  }

  /* insert the new element */
  info[i+1].score          = score;
  info[i+1].ref_id         = ref_id;
  info[i+1].ref_start      = (ref_pos + alignment->best_j0);
  info[i+1].ref_end        = (ref_pos + alignment->best_j);
  info[i+1].read_start     = (alignment->best_i0);
  info[i+1].read_end       = (alignment->best_i);
  info[i+1].sense          = (alignment_sense & ALIGNMENT_SENSE_REVERSE ? (-1) : 1);
  SAFE_FAILURE__ALLOC(info[i+1].traceback, TRACEBACK_SEQ_SIZE(alignment->traceback_seq_len), unsigned char);
  info[i+1].traceback_seq_len = alignment->traceback_seq_len;
  /* copy but reverse the traceback of the alignment ("right order" now) */
  int p;
  for (p = 0; p < alignment->traceback_seq_len; ++p) {
    TRACEBACK_SEQ_SET(info[i+1].traceback, p, TRACEBACK_SEQ_GET(alignment->traceback, (alignment->traceback_seq_len - 1 - p )));
  }

#ifdef _OPENMP
#pragma omp atomic
#endif
  map->mapped += mapped;
  return i+1;
}

/**
 * Obtain the score for a read, when mapped on a given position
 * @param map
 * @param read_id The read index
 * @param ref_id The reference index
 * @param ref_start The position of interest in the reference genome
 * @param rank The rank of that score will be placed at this address
 * @return The score
 */
inline int hit_map__score_for(const HitMapType* map, const int read_id, const int ref_id, const int ref_start) {
  if (map->map[read_id]) {
    int k;
    for (k = 0; k < MAP_DETAIL_SIZE; ++k) {
      if ( map->map[read_id][k].ref_id == ref_id  &&  map->map[read_id][k].ref_start == ref_start) {
        return map->map[read_id][k].score;
      }
    }
  }
  return MIN_SCORE;
}

/**
 * Quick display of the successfully matched reads: index, position on the reference, score, adjusted_score [FIXME] and traceback
 * @param map
 */
void hit_map__display(const HitMapType* map) {
  int i;
  for (i = 0; i < map->size; ++i) {
    if (map->map[i]) {
      int k;
      for (k = 0; k < MAP_DETAIL_SIZE && map->map[i][k].ref_start != UNMAPPED; ++k) {
        printf("read:%d:[%d,%d]  ref:%d:[%d-%d]  score: %d  adj_score: %d  len: %d\n", i, map->map[i][k].read_start, map->map[i][k].read_end, map->map[i][k].ref_id, map->map[i][k].ref_start, map->map[i][k].ref_end, map->map[i][k].score, map->map[i][k].adjusted_score, map->map[i][k].traceback_seq_len);
        TRACEBACK_DISPLAY(map->map[i][k].traceback, map->map[i][k].traceback_seq_len, stdout);
        printf("\n");
      }
    }
  }
}


/**
 * Output the hit_map data as a full SAM output (avoid the "GenomeMap" that keep the "best read" per "genome pos" ; here display the score_rank_max best matching per read)
 * @param map
 * @param reads_db
 * @param ref_dbs
 * @param sam_output
 * @param score_rank_max is the maximal number of position mapped per read (cannot be more than MAP_DETAIL_SIZE)
 */
extern char** cmd_line;
extern int    cmd_line_len;

void hit_map__generate_SAM_output(const HitMapType* map,
                                  const ReadsDBType* reads_db,
                                  const ReferenceDBType* ref_dbs, const int ref_dbs_size,
                                  FILE* sam_output, const int score_rank_max) {

  /* Worst case: 1M1I1M1I ... => 4*read length [FIXME] : size of the matches runs (101M1IM1I) should be take into account + maximal number of indels should also be measured ... */
  int   CIGAR_max_size = reads_db->read_len << 2;
  char* CIGAR_STRING;

#ifdef NUCLEOTIDES
  int translated_read_max_size = reads_db->read_len + 1;
#else
  int translated_read_max_size = reads_db->read_len + 2;
  CODE_TYPE ** ref_dbs_symbol  = NULL;
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

#ifndef NUCLEOTIDES
  /* build the ref_dbs_symbol to translate into nucleotide code "unsorted" reads */
  SAFE_FAILURE__ALLOC(ref_dbs_symbol, ref_dbs_size, CODE_TYPE*);
  {
    int ref_id;
    for (ref_id = 0; ref_id < ref_dbs_size; ++ref_id) {
      SAFE_FAILURE__ALLOC(ref_dbs_symbol[ref_id], COMPRESSED_LEN(ref_dbs[ref_id].size + 1), CODE_TYPE);
      int ref_pos = 0;
      CODE_TYPE ref_symbol = ref_dbs[ref_id].first_base;
      for (ref_pos = 0 ; ref_pos < ref_dbs[ref_id].size ; ++ref_pos) {
        TO_NTH_CODE(ref_dbs_symbol[ref_id], ref_pos, ref_symbol);
        ref_symbol = TRANSFORM(ref_symbol, NTH_CODE(ref_dbs[ref_id].sequence, ref_pos));
      }
      TO_NTH_CODE(ref_dbs_symbol[ref_id], ref_pos, ref_symbol);
    }
  }
#endif
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

  /* SAM Version:1.4, Sorting Order:unsorted */
  fprintf(sam_output, "@HD\tVN:1.4\tSO:unsorted\n");

  /* Header section first */
  {
    int ref_id;
    for (ref_id = 0; ref_id < ref_dbs_size; ++ref_id) {
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
  QUAL  | [!-~]+|\*             | [0,93]        | query QUALity; ASCII-33 (by default) the Phred base quality
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

  int read_id;
  for (read_id = 0; read_id < map->size; ++read_id) {
    if (map->map[read_id]) {
      ReadDataType* read = &(reads_db->reads[read_id]);
      int score_rank;
      int nb_reported = 0;

      /* count number of reported */
      for (score_rank = 0; score_rank < score_rank_max && map->map[read_id][score_rank].ref_start != UNMAPPED && ((UNIQUENESS == 0) || ((score_rank + 1 < score_rank_max) && ((map->map[read_id][score_rank].score - map->map[read_id][score_rank + 1].score) >= UNIQUENESS_SCORE_THRESHOLD)) || (score_rank + 1 == score_rank_max)); ++score_rank) {
        nb_reported++;
      }
      /* then output */
      for (score_rank = 0; score_rank < score_rank_max && map->map[read_id][score_rank].ref_start != UNMAPPED && ((UNIQUENESS == 0) || ((score_rank + 1 < score_rank_max) && ((map->map[read_id][score_rank].score - map->map[read_id][score_rank + 1].score) >= UNIQUENESS_SCORE_THRESHOLD)) || (score_rank + 1 == score_rank_max)); ++score_rank) {
        int flag           = FLAG_EMPTY;
        if (map->map[read_id][score_rank].sense & ALIGNMENT_SENSE_REVERSE) {
          flag |= FLAG_REVERSE;
        }
        if (score_rank && (map->map[read_id][score_rank].score != map->map[read_id][0].score)) {
          flag |= FLAG_NOT_PRIMARY;
        }
        int ref_id =
          map->map[read_id][score_rank].ref_id;
        int ref_start =
          map->map[read_id][score_rank].ref_start;

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
                                                          ref_dbs[ref_id].sequence + COMPRESSED_IDX(ref_start), COMPRESSED_OFFSET(ref_start),
#ifndef NUCLEOTIDES
                                                          NTH_CODE(ref_dbs_symbol[ref_id],ref_start),
#endif
                                                          CIGAR_STRING, translated_read_string);

        /* Read name, flag, reference name, mapping position, alignment quality, CIGAR string */
        fprintf(sam_output, "%s\t%d\t%s\t%d\t%d\t%s\t", read->info, flag, DISPLAY_NAME(ref_dbs[ref_id]), (ref_start+1) /* genome positions for 1 to N */, alignment_quality, CIGAR_STRING);
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
              fprintf(sam_output, "%c", read_quality_min_symbol_code + READ_QUALITY_LEVEL_UPPER_BOUNDS[NTH_QUAL(reads_db->reads[read_id].quality, q)]);
            }
#ifndef NUCLEOTIDES
            fprintf(sam_output, "%c", read_quality_min_symbol_code + READ_QUALITY_LEVEL_UPPER_BOUNDS[reads_db->reads[read_id].first_qual]);
#endif
          } else {
            int q;
#ifndef NUCLEOTIDES
            fprintf(sam_output, "%c", read_quality_min_symbol_code + READ_QUALITY_LEVEL_UPPER_BOUNDS[reads_db->reads[read_id].first_qual]);
#endif
            for (q = 0; q < reads_db->read_len; ++q) {
              fprintf(sam_output, "%c", read_quality_min_symbol_code + READ_QUALITY_LEVEL_UPPER_BOUNDS[NTH_QUAL(reads_db->reads[read_id].quality, q)]);
            }
          }
        } else {
          /* quality replaced by a "*" if no quality */
          fprintf(sam_output, "*");
        }
        fprintf(sam_output, "\tAS:i:%d\tNM:i:%d\tNH:i:%d\n", map->map[read_id][score_rank].score, edit_distance, nb_reported);
      }
    } else {
      /* Unmapped read case (for v1.5) : not checked yet */
      /*
       * ReadDataType* read = &(reads_db->reads[read_id]);
       */
      /*
       * Read name, flag, reference name, mapping position, alignment quality, CIGAR string, sequence, sequence quality
       * The mate information (mate reference name, mate position, inserted size) is not supported yet (TODO) [FIXME]
       */
      /*
       * fprintf(sam_output, "%s\t4\t*\t0\t255\t*\t*\t*\t=\t0\t0\n", read->info);
       */
    }
  }
  fflush(sam_output);
  free(CIGAR_STRING);
  free(translated_read_string);

#ifndef NUCLEOTIDES
  /* free the ref_dbs_symbol */
  {
    int ref_id;
    for (ref_id = 0; ref_id < ref_dbs_size; ++ref_id)
      free(ref_dbs_symbol[ref_id]);
    free(ref_dbs_symbol);
  }
#endif
}



/**
 * Cleanup
 * @param map
 */
void hit_map__destroy(HitMapType* map) {
  int i;
  for (i = 0; i < map->size; ++i) {
    if (map->map[i]) {
      int k;
      for (k = 0; k < MAP_DETAIL_SIZE; ++k) {
        if (map->map[i][k].traceback) {
          free(map->map[i][k].traceback);
          map->map[i][k].traceback = NULL;
        }
      }
      free(map->map[i]);
      map->map[i] = NULL;
    }
  }
  free(map->map);
  map->map = NULL;
}
