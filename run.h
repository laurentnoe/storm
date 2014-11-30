#ifndef _RUN_H_
#define _RUN_H_

#include "alignment.h"
#include "alignment_simd.h"
#include "index.h"
#include "load_data.h"
#include "genome_map.h"

int single_alignment(const char* read_str, const char* ref_str, const char* ref_masked_str, const char* qual_str,
                     const ScoreType match, const ScoreType mismatch, const ScoreType gap_open, const ScoreType gap_extend, const int allowed_indels,
                     FILE* output);

int reads_against_references(const char* reads_filename, const char* qual_filename, const char* ref_filename,
                             const char* seeds,
                             const ScoreType match, const ScoreType mismatch, const ScoreType gap_open, const ScoreType gap_extend, const int allowed_indels, const int simd_allowed_diags,
                             FILE* output, FILE* unmapped_FASTQ_output);

#endif /* _RUN_H_ */
