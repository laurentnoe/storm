#include "stats.h"

/**
 * Displays the traceback strings of the mapped reads (for statistics)
 */
void  display_tracebacks(GenomeMapType* genome_map) {
  static int traceback_patterns_index = 1;
  static char filename[256];
  sprintf(filename, "traceback_patterns%05d.txt", traceback_patterns_index);
  FILE* f = fopen(filename, "w");
  int ref_id;
  /* reference order */
  for (ref_id = 0; ref_id < genome_map->ref_dbs_size; ++ref_id) {
    int ref_pos;
    for (ref_pos = 0; ref_pos < genome_map->ref_dbs[ref_id].size; ++ref_pos) {
      if (genome_map->g_maps[ref_id][ref_pos]) {
      int k;
      for (k = 0; k < genome_map->g_maps[ref_id][ref_pos]->read_at_pos_size; ++k) {
        int read_id    = genome_map->g_maps[ref_id][ref_pos]->read_at_pos[k].read_idx;
        int score_rank = genome_map->g_maps[ref_id][ref_pos]->read_at_pos[k].score_rank;
        TRACEBACK_DISPLAY((genome_map->hitmap->map[read_id][score_rank].traceback), genome_map->hitmap->map[read_id][score_rank].traceback_seq_len, f);
        fprintf(f, "\t%d", genome_map->hitmap->map[read_id][score_rank].score);
        if (genome_map->hitmap->map[read_id][score_rank].sense & (ALIGNMENT_SENSE_REVERSE)) {
          fprintf(f, "\tR");
        }
        fprintf(f, "\n");
      }
      }
    }
  }
  fflush(f);
  fclose(f);
  ++traceback_patterns_index;
}

/**
 * Lists the unmapped reads,
 */
void list_unmapped_reads_translated(GenomeMapType* genome_map) {
  FILE* f = fopen("unmapped_reads_translated.txt", "w");
  int read_id;
  for (read_id = 0; read_id < genome_map->reads_db->size; ++read_id) {
    if (genome_map->hitmap->map[read_id][0].score == MIN_SCORE) {
      if (genome_map->reads_db->reads[read_id].info) {
        fprintf(f, ">%s\n", genome_map->reads_db->reads[read_id].info);
      } else {
        fprintf(f, ">%d\n", read_id);
      }
#ifdef NUCLEOTIDES
      {
        int k;
        for (k = 0; k < genome_map->reads_db->read_len; ++k) {
          fprintf(f, "%c", BASE_CODE_LETTER[NTH_CODE(genome_map->reads_db->reads[read_id].sequence, k)]);
        }
        fprintf(f, "\n");
      }
#else
      {
      CODE_TYPE base = genome_map->reads_db->reads[read_id].first_base;
      fprintf(f, "%c", BASE_CODE_LETTER[base]);
      int k;
        for (k = 0; k < genome_map->reads_db->read_len; ++k) {
        base = COMPOSE(base,NTH_CODE(genome_map->reads_db->reads[read_id].sequence,k));
          fprintf(f, "%c", BASE_CODE_LETTER[base]);
        }
        fprintf(f, "\n");
      }
#endif
    }
  }
  fflush(f);
  fclose(f);
}

void list_unmapped_reads(GenomeMapType* genome_map) {
  FILE* f = fopen("unmapped_reads.txt", "w");
  int read_id;
  for (read_id = 0; read_id < genome_map->reads_db->size; ++read_id) {
    if (genome_map->hitmap->map[read_id][0].score == MIN_SCORE) {
      if (genome_map->reads_db->reads[read_id].info) {
        fprintf(f, ">%s\n", genome_map->reads_db->reads[read_id].info);
      } else {
        fprintf(f, ">%d\n", read_id);
      }
#ifdef NUCLEOTIDES
      {
        int k;
        for (k = 0; k < genome_map->reads_db->read_len; ++k) {
          fprintf(f, "%c", BASE_CODE_LETTER[NTH_CODE(genome_map->reads_db->reads[read_id].sequence, k)]);
        }
        fprintf(f, "\n");
      }
#else
      {
      fprintf(f, "%c", BASE_CODE_LETTER[genome_map->reads_db->reads[read_id].first_base]);
      int k;
        for (k = 0; k < genome_map->reads_db->read_len; ++k) {
          fprintf(f, "%d",NTH_CODE(genome_map->reads_db->reads[read_id].sequence,k));
        }
        fprintf(f, "\n");
      }
#endif
    }
  }
  fflush(f);
  fclose(f);
}
