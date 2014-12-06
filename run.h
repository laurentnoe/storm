#ifndef _RUN_H_
#define _RUN_H_

#include "alignment.h"
#include "alignment_simd.h"
#include "index.h"
#include "load_data.h"
#include "genome_map.h"

/**
 * Perform an alignment between two sequences
 * @param read_str First sequence (color/nucleotide uncompressed code)
 * @param ref_str Second sequence (color/nucleotide uncompressed code)
 * @param ref_masked_str Second sequence mask (uncompressed code)
 * @param read quality values for the first sequence
 * @param match
 * @param mismatch
 * @param gap_open
 * @param gap_extend
 * @param allowed_indels
 * @param output The output file descriptor
 */
int single_alignment(const char* read_str, const char* ref_str, const char* ref_masked_str, const char* qual_str,
                     const ScoreType match, const ScoreType mismatch, const ScoreType gap_open, const ScoreType gap_extend, const int allowed_indels,
                     FILE* output);

/**
 * Reads mapped on a reference
 * @param reads_filename
 * @param qual_filename
 * @param ref_filename
 * @param seed_list
 * @param match
 * @param mismatch
 * @param gap_open
 * @param gap_extend
 * @param allowd_indels
 * @param output
 * @param unmapped_FASTQ_output
 */
int reads_against_references(const char* reads_filename, const char* qual_filename, const char* ref_filename,
                             const char* seeds,
                             const ScoreType match, const ScoreType mismatch, const ScoreType gap_open, const ScoreType gap_extend, const int allowed_indels, const int simd_allowed_diags,
                             FILE* output, FILE* unmapped_FASTQ_output);

#endif /* _RUN_H_ */
