#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "util.h"
#include "run.h"
#include "alignment_simd.h"

char** cmd_line;
int    cmd_line_len;

extern char *optarg;

extern ScoreType min_accepted_score;
extern ScoreType min_accepted_score_simd_filter;

extern int map_greedy;
extern int map_unordered;

extern int SHOW_TRACEBACK_PATTERNS;
extern int LIST_UNMAPPED_READS;

extern int GENERATE_KMER_HISTOGRAM;
extern int KMER_HISTOGRAM_INTERVAL_WIDTH;
extern int FILTER_REPEATS;
extern int ACCEPTED_STDEV_DISTANCE;
extern int IGNORE_LOWERCASE;
extern int UNIQUENESS;
extern int UNIQUENESS_SCORE_THRESHOLD;
extern int PRINT_ALL_READS;

extern int read_quality_min_symbol_code;

       int allowed_indels      = DEFAULT_ALLOWED_INDELS;
       int simd_allowed_diags  = DEFAULT_ALLOWED_INDELS;
       int delta_m;

#define DISPLAY_OPTION(option,...) {MESSAGE__("\t\033[37;1m-%s\033[0m: ", option); MESSAGE__(__VA_ARGS__) ; MESSAGE__("\n");}

#define INTEGER_VALUE_FROM_OPTION(target) {                                     \
    if (sscanf(optarg, "%d", &target) != 1) {                                   \
      ERROR__("Mandatory numeric parameter missing for option %c.", option);    \
      exit(1);                                                                  \
    }                                                                           \
  }

#define STRING_VALUE_FROM_OPTION(target) {                                                                             \
    if ((strlen(optarg) == 2) && (optarg[0] == '-') && (optarg[1] != ':') && (strchr(optstring, optarg[1]) != NULL)) { \
      ERROR__("Mandatory parameter missing for option %c.", option);                                                   \
      exit(1);                                                                                                         \
    } else {                                                                                                           \
      target = optarg;                                                                                                 \
    }                                                                                                                  \
  }

#define HANDLE_INVALID_NUMERIC_VALUE(value, default_) {                                              \
    WARNING__("Invalid value %d for option %c. Resetting default (%d).", value, option, default_);   \
    value = default_;                                                                                \
  }

#define HANDLE_INVALID_NUMERIC_VALUE_FATAL(value, expected) {                            \
    ERROR__("Invalid value %d for option %c. Expected: %s.", value, option, expected);   \
    exit(1);                                                                             \
  }


static void show_usage(char * progname, char* default_seeds) {
  MESSAGE__("\n%s (beta v%s)\n\n", PROGRAM_NAME, PROGRAM_VERSION);
  MESSAGE__("\nUSAGE: %s parameters\n\n",progname);
  MESSAGE__("PARAMETERS:\n");
  MESSAGE__("\033[37;1mMandatory:\033[0m\n");
  DISPLAY_OPTION("g <genome_file>", "Name of the reference file (.fasta/.multifasta).\n");
  DISPLAY_OPTION("r <reads_file>",  " Name of the reads file (.csfasta/.fastq/.multifasta)." );
  /* [FIXME] :
   */
  /*
   * (.csfasta\n"
   * "\t   will be automatically appended). The quality file with the same\n"
   * "\t   name and the .qual extension will be considered,  if it exists.\n"));
   */
  /*
   * Hiding options that are very unlikely to be used in practice
   */
  /*
   * MESSAGE__("or\n");
   * DISPLAY_OPTION("f <color sequence>", "Reference sequence.  To be used with  -a and\n"
   *                "\t   optionally -q,  when just the alignment of two  small sequences\n"
   *                "\t   is wanted, instead of -r and -g.\n");
   * DISPLAY_OPTION("a <color sequence>", "Query sequence. To be used with -f and optionally\n"
   *                "\t   -p,  when just  the alignment of two  small sequences  is wanted,\n"
   *                "\t   instead of -r and -g.\n");
   */
  MESSAGE__("\n\t\033[33;1mWARNING\033[0m: By default,  at most  one read is kept  per reference position.\n"
            "\t         This can be changed with the  -M <number>  parameter  (possibly\n"
            "\t         adding the -A parameter to output unmapped reads)\n\n");
  MESSAGE__("\n\033[37;1mOptional:\033[0m\n");
  DISPLAY_OPTION("q <qualities_file>",  "Name of the qualities file (qual).\n");
  DISPLAY_OPTION("s <seed list>", "The  seed family  to be used for filtering. It should be\n"
                 "\t   of the form \"#-#-#;##--#\" or \"10101;11001\"  (seeds separated by ';' or '\\;').\n"
                 "\t   These seeds are applied on every position of the read. To restrict to\n"
                 "\t   certain positions, the list of wanted positions w.r.t the read can\n"
                 "\t   be mentioned after each seed, as follows: \n"
                 "\t           \"#-#-#:0 3 5;##--#:1 2 4 6\".\n"
                 "\t   Default seeds:  %s\n", default_seeds);
  DISPLAY_OPTION("S <seed file>",  "The file containing the  seeds to be used for filtering,\n"
                 "\t   int a format similar to the one described for the -s <seed list>. For\n"
                 "\t   more clarity,  the seeds can be separated by  '\\n' instead of  ';' in\n"
                 "\t   the file. Default seed file: %s\n", DEFAULT_SEED_FILE);
  DISPLAY_OPTION("l <number>",  "This parameter controls the removal of very frequent k-mers\n"
                 "\t   from the index. The  numeric parameter establishes which  k-mers will\n"
                 "\t   be erased:  it expresses the  acceptable distance  from  the  average\n"
                 "\t   number of occurrences,  measured in number of standard deviations. By\n"
                 "\t   default, the value for this parameter is %3d.  To disable this, use a\n"
                 "\t   negative value.\n", DEFAULT_ACCEPTED_STDEV_DISTANCE);
  DISPLAY_OPTION("j", "Ignore lowercase symbols from the reference sequences.\n");
  /*
   * Hiding options that are very unlikely to be used in practice
   */
  /*
   * DISPLAY_OPTION("p <integer sequence>", "Read quality sequence. To be used with -a and \n"
   *                "\t   -f, when just the alignment of two small sequences is wanted, \n"
   *                "\t    instead of -r and -g.\n");
   */
  DISPLAY_OPTION("F", "Only align in forward sense.                    Default: both senses\n");
  DISPLAY_OPTION("R", "Only align in reverse complementary sense.      Default: both senses\n");
  DISPLAY_OPTION("o <output_file>", "Name of the  output file  where the  mapping result is\n"
                 "\t   written, in SAM format.                               Default: stdout\n");
  DISPLAY_OPTION("O <output_file>", "Name of the  output file  where the unmapped reads are\n"
                 "\t   written, in FASTQ format. Availaible if -M <number>.    Default: none\n");
  DISPLAY_OPTION("m <number>", "Match score.                                   Default: %3d\n", SCALE_SCORE(DEFAULT_MATCH, MAX_READ_QUALITY_LEVEL));
  DISPLAY_OPTION("x <number>", "Mismatch penalty.                              Default: %3d\n"
                 "\n\t\033[33;1mWARNING\033[0m: Match / mismatch scores are \033[37;1mscaled\033[0m according to read qualities,"
                 "\n\t         if qualities are available. Please, see the -b <number> option.\n", SCALE_SCORE(DEFAULT_MISMATCH, MAX_READ_QUALITY_LEVEL));
  DISPLAY_OPTION("d <number>", "Gap open penalty.                              Default: %3d\n", DEFAULT_GAP_OPEN);
  DISPLAY_OPTION("e <number>", "Gap extension penalty.                         Default: %3d\n", DEFAULT_GAP_EXTEND);
  DISPLAY_OPTION("i <number>", "Maximum number of indels in the alignment.     Default: %3d\n", DEFAULT_ALLOWED_INDELS);
  DISPLAY_OPTION("t <number>", "Minimum accepted score, above which alignments are selected\n"
                 "\t   because considered significant.                          Default: %3d\n", MIN_ACCEPTED_SCORE);
  DISPLAY_OPTION("z <number>", "The minimum score accepted by the SIMD filter,  above which\n"
                 "\t   alignments are selected.                                 Default: %3d\n", MIN_ACCEPTED_SCORE_SIMD_FILTER);
  DISPLAY_OPTION("u <number>", "The  minimum difference  between  the first and  the second\n"
                 "\t   alignment of a given read to consider it unique;    Default: disabled\n");
  DISPLAY_OPTION("b <number>", "The ASCII code for the  worst possible quality value of the\n"
                 "\t   FASTQ file, given in decimal format.                Default: %2d (\'%c\')", read_quality_min_symbol_code, (char) read_quality_min_symbol_code);
  MESSAGE__("\n\t\033[33;1mWARNING\033[0m: Match / mismatch scores are scaled according to read qualities.\n"
            "\t         Setting this parameter to 0 DISABLES this scaling behaviour.\n\n");
  DISPLAY_OPTION("G", "Perform an \"ordered greedy\" mapping. By default, multiple alignments\n"
                 "\t    of overlapping reads are evaluated during the mapping stage in order\n"
                 "\t    to establish the  most relevant candidate read for each  position of\n"
                 "\t    the reference. If this option is present, only the score of the read\n"
                 "\t    alignment is taken into account.\n");
  DISPLAY_OPTION("M <number>", "Perform an  \"unordered\" mapping. By default/in greedy mode,\n"
                 "\t    only one read  is kept per reference position.  This option  enables\n"
                 "\t    any read to be mapped at any number (from 1 to %d) positions per read\n"
                 "\t    without considering that this read is necessary the best scoring one\n"
                 "\t    from the reference point of view.\n"
                 "\n\t\033[33;1mWARNING\033[0m: Not all mappable reads \033[37;1mwill be diplayed\033[0m WITHOUT this option set\n",
                 MAP_DETAIL_SIZE);
  DISPLAY_OPTION("A", "Output unmapped reads in the SAM output file.  This is only possible\n"
                 "\t    when the -M <number> \"unordered mapping\" is activated.\n");
#ifdef _OPENMP
  DISPLAY_OPTION("N <number>", "Limit the number of OpenMP threads.\n");
#endif
  DISPLAY_OPTION("v <number>", "Verbose - output message detail level.\n"
                 "\t   %1d: no message, %1d: moderate, %d: high, %1d: exaggerate.       Default: %2d\n",
                 VERBOSITY_NONE, VERBOSITY_MODERATE, VERBOSITY_HIGH, VERBOSITY_ANNOYING, VERBOSITY_DEFAULT);
  DISPLAY_OPTION("h", "Displays usage and exits.\n");
}


int main (int argc, char* argv[]) {
  /*
   * Possible options:
   * -g genome_file
   * -r reads_file (without extension; implicit extension: .fa; implicit quality file -- if any -- .qual)
   * -a read, as a color sequence;
   * -q read quality, as an integer sequence;
   * -f reference  color sequence;
   * -m match score
   * -x mismatch penalty
   * -d gap open penalty
   * -e gap extension penalty
   * -i number of allowed indels in the alignment
   */
  char *optstring = "g:r:q:s:S:a:b:f:p:m:x:d:e:t:z:T:i:v:o:O:l:u:M:N:jFRhcPHUGA";
  /* Input files */
  char *genome_file = NULL, *reads_file = NULL, *qual_file = NULL;
  char *read = NULL, *reference=NULL, *reference_masked=NULL, *quality=NULL;
  char *seeds, *tmp;
  /* Alignment parameters, initialized with the default values from alignment.h */
  ScoreType match = DEFAULT_MATCH, mismatch = DEFAULT_MISMATCH, gap_open = DEFAULT_GAP_OPEN, gap_extend = DEFAULT_GAP_EXTEND;

  /* Output files*/
  FILE* output = stdout;
  /* Fastq Ouput for missing reads */
  FILE* unmapped_FASTQ_output = NULL;

  /* Process command line ... */
  int option;

#ifdef __AVX512BW__
  if (!alignment_avx512bw__compatible_proc()) {ERROR__("CPU is not compatible with AVX512BW instructions set.\nExiting.\n"); exit(1);}
#endif
#ifdef __AVX2__
  if (!alignment_avx2__compatible_proc()) {ERROR__("CPU is not compatible with AVX2 instructions set.\nExiting.\n"); exit(1);}
#endif
#ifdef __SSE2__
  if (!alignment_sse2__compatible_proc()) {ERROR__("CPU is not compatible with SSE2 instructions set.\nExiting.\n"); exit(1);}
#endif
#ifdef __SSE__
  if (!alignment_sse__compatible_proc())  {ERROR__("CPU is not compatible with SSE instructions set.\nExiting.\n");  exit(1);}
#endif

  cmd_line = argv;
  cmd_line_len = argc;

  seeds = seed__string_from_file(DEFAULT_SEED_FILE);
  if (!seeds) {
    seeds = DEFAULT_SEEDS;
  }

#ifdef _OPENMP
  MAX_THREADS = omp_get_num_procs();
#endif

  while((option  = getopt(argc, argv, optstring)) != -1) {
    switch (option) {
    case 'g':
      if (genome_file != NULL) {
        ERROR__("Genome file previously set with \"%s\".\nUnable to set it again.\n", genome_file);
        exit(1);
      }
      STRING_VALUE_FROM_OPTION(genome_file);
      break;
    case 'r':
      if (reads_file != NULL) {
        ERROR__("Read file previously set with \"%s\".\nUnable to set it again.\n", reads_file);
        exit(1);
      }
      STRING_VALUE_FROM_OPTION(reads_file);
      break;
    case 'q':
      if (qual_file != NULL) {
        ERROR__("Quality file previously set with \"%s\".\nUnable to set it again.\n", qual_file);
        exit(1);
      }
      STRING_VALUE_FROM_OPTION(qual_file);
      break;
    case 'S':
      if ((tmp = seed__string_from_file(optarg)) == NULL) {
        char confirmation;
        ERROR__("Unable to read seeds input file %s.", optarg);
        do {
          MESSAGE__("Continue (c) with default seeds %s, or abort (a)?\n", seeds);
        } while(scanf("%c", &confirmation) && confirmation != 'a' && confirmation != 'A' && confirmation != 'c' && confirmation != 'C');
        if (confirmation == 'a' || confirmation == 'A') {
          exit(1);
        }
      } else {
        seeds = tmp;
      }
      break;
    case 's':
      seeds = optarg;
      break;
    case 'm':
      INTEGER_VALUE_FROM_OPTION(match);
      if (match <= 0) {
        HANDLE_INVALID_NUMERIC_VALUE_FATAL(match, "a strictly positive integer");
      }
      break;
    case 'x':
      INTEGER_VALUE_FROM_OPTION(mismatch);
      if (mismatch > 0) {
        HANDLE_INVALID_NUMERIC_VALUE_FATAL(mismatch, "a negative value or zero");
      }
      break;
    case 'd':
      INTEGER_VALUE_FROM_OPTION(gap_open);
      if (gap_open > 0) {
        HANDLE_INVALID_NUMERIC_VALUE_FATAL(gap_open, "a negative value or zero");
      }
      break;
    case 'e':
      INTEGER_VALUE_FROM_OPTION(gap_extend);
      if (gap_extend > 0) {
        HANDLE_INVALID_NUMERIC_VALUE_FATAL(gap_extend, "a negative value or zero");
      }
      break;
    case 't':
      INTEGER_VALUE_FROM_OPTION(min_accepted_score);
      break;
    case 'z':
      INTEGER_VALUE_FROM_OPTION(min_accepted_score_simd_filter);
      break;
    case 'b':
      INTEGER_VALUE_FROM_OPTION(read_quality_min_symbol_code);
      break;
    case 'i':
      INTEGER_VALUE_FROM_OPTION(allowed_indels);
      if (allowed_indels < 0) {
        HANDLE_INVALID_NUMERIC_VALUE_FATAL(allowed_indels, "a positive value or zero");
      }
      if (allowed_indels >= INDEL_DATA_VECTOR_SIZE) {
        VERB_FILTER(VERBOSITY_MODERATE, WARNING__("%s may not handle more than %d indels in SIMD (local alignment) filter (%d diagonals).\nHowever final alignment does support this -i <%d> value...\nSo a good idea is to set -z <number> with smaller value than -t <number>.", PROGRAM_NAME, INDEL_DATA_VECTOR_SIZE-1, INDEL_DATA_VECTOR_SIZE, allowed_indels));
        simd_allowed_diags = INDEL_DATA_VECTOR_SIZE-1;
      } else {
        simd_allowed_diags = allowed_indels;
      }
      break;
    case 'a':
      read = optarg;
      break;
    case 'p':
      quality = optarg;
      break;
    case 'f':
      reference = optarg;
      reference_masked = malloc(strlen(optarg) + 1);
      memset(reference_masked,'A',strlen(optarg)); /* create an empty mask of smae size */
      reference_masked[strlen(optarg)] = '\0';
      break;
    case 'o':
      if (strcmp(optarg, "NULL") == 0 || strcmp(optarg, "null") == 0) {
        output = NULL;
      } else {
        output = fopen(optarg, "w");
        if (output == NULL) {
          char confirmation;
          ERROR__("Could not open or create output file %s.\n", optarg);
          do {
            MESSAGE__("Continue (c) and output to stdout, or abort (a)?\n");
          } while(scanf("%c", &confirmation) && confirmation != 'a' && confirmation != 'A' && confirmation != 'c' && confirmation != 'C');
          if (confirmation == 'c' || confirmation == 'C') {
            output = stdout;
          } else {
            exit(1);
          }
        }
      }
      break;
    case 'O':
      if (strcmp(optarg, "NULL") == 0 || strcmp(optarg, "null") == 0) {
        unmapped_FASTQ_output = NULL;
      } else {
        unmapped_FASTQ_output = fopen(optarg, "w");
        if (unmapped_FASTQ_output == NULL) {
          ERROR__("Could not open or create FASTQ output file %s.\n", optarg);
          exit(1);
        }
      }
      break;
    case 'F':
      ALIGNMENT_SENSE = ALIGNMENT_SENSE_FORWARD;
      break;
    case 'R':
      ALIGNMENT_SENSE = ALIGNMENT_SENSE_REVERSE;
      break;
    case 'h':
      show_usage(argv[0], seeds);
      return 0;
    case 'G':
      map_greedy = 1;
      break;
    case 'M':
      INTEGER_VALUE_FROM_OPTION(map_unordered);
      if (map_unordered <= 0) {
        HANDLE_INVALID_NUMERIC_VALUE_FATAL(map_unordered, "a strictly positive integer");
      }
      if (map_unordered > MAP_DETAIL_SIZE) {
        HANDLE_INVALID_NUMERIC_VALUE_FATAL(map_unordered, "a value smaller than MAP_DETAIL_SIZE (4)");
      }
      break;
    case 'A':
      if (!map_unordered) {
         ERROR__("The -A (output all reads in SAM) parameter is not available with the Ordered Mapping\nbecause not all mappable reads are already displayed in this mode.\nPlease use the Unordered Mapping (-M <number>) before using -A.\n");
         exit(1);
      }
      PRINT_ALL_READS = 1;
      break;

    case 'v':
      INTEGER_VALUE_FROM_OPTION(VERBOSITY);
      if (VERBOSITY < VERBOSITY_NONE || VERBOSITY > VERBOSITY_MAX) {
        HANDLE_INVALID_NUMERIC_VALUE(VERBOSITY, VERBOSITY_DEFAULT);
      }
      break;
    case 'P':
      SHOW_TRACEBACK_PATTERNS = 1;
      break;
    case 'H':
      GENERATE_KMER_HISTOGRAM = 1;
      INTEGER_VALUE_FROM_OPTION(KMER_HISTOGRAM_INTERVAL_WIDTH);
      if (KMER_HISTOGRAM_INTERVAL_WIDTH <= 0) {
        HANDLE_INVALID_NUMERIC_VALUE(KMER_HISTOGRAM_INTERVAL_WIDTH, 1);
      }
      break;
#ifdef _OPENMP
    case 'N':
      INTEGER_VALUE_FROM_OPTION(MAX_THREADS);
      if (MAX_THREADS <= 0) {
        HANDLE_INVALID_NUMERIC_VALUE_FATAL(MAX_THREADS, "a strictly positive integer");
      }
      break;
#endif
    case 'U':
      LIST_UNMAPPED_READS = 1;
      break;
    case 'l':
      FILTER_REPEATS = 1;
      INTEGER_VALUE_FROM_OPTION(ACCEPTED_STDEV_DISTANCE);
      if (ACCEPTED_STDEV_DISTANCE <= 0) {
        FILTER_REPEATS = 0;
        ACCEPTED_STDEV_DISTANCE = 0;
      }
      break;
    case 'j':
      IGNORE_LOWERCASE = 1;
      break;
    case 'u':
      UNIQUENESS = 1;
      INTEGER_VALUE_FROM_OPTION(UNIQUENESS_SCORE_THRESHOLD);
      if (UNIQUENESS_SCORE_THRESHOLD <= 0) {
        HANDLE_INVALID_NUMERIC_VALUE_FATAL(UNIQUENESS_SCORE_THRESHOLD, "a positive integer");
      }
      break;
    default:
      exit(1);
    }
  }

  delta_m = match - mismatch;

  if (read && reference) {
    VERB_FILTER(VERBOSITY_MODERATE, MESSAGE__("\nTwo sequence alignment:\n%s\n%s\n", read, reference););
    if (quality) {
      VERB_FILTER(VERBOSITY_MODERATE, MESSAGE__("%s\n", quality););
    }
    single_alignment(read, reference, reference_masked, quality, match, mismatch, gap_open, gap_extend, allowed_indels, output);
  } else if (reads_file) {
    if (genome_file) {
      VERB_FILTER(VERBOSITY_MODERATE, MESSAGE__("\n\nAligning reads against reference.\nInput files: %s, %s, %s.\n\n", genome_file, reads_file, qual_file?qual_file:"(null)"););
      reads_against_references(reads_file, qual_file, genome_file, seeds, match, mismatch, gap_open, gap_extend, allowed_indels, simd_allowed_diags, output, unmapped_FASTQ_output);
    } else {
      ERROR__("\nNot enough mandatory parameters provided.\n");
      show_usage(argv[0], seeds);
    }
  } else {
    ERROR__("\nNot enough mandatory parameters provided.\n");
    show_usage(argv[0], seeds);
  }

  if (output && output != stdout) {
    fclose(output);
  }
  if (unmapped_FASTQ_output) {
    fclose(unmapped_FASTQ_output);
  }
  return 0;
}
