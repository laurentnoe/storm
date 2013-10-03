#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "run.h"

char** cmd_line;
int    cmd_line_len;

extern char *optarg;
extern int optind, opterr, optopt;

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

int allowed_indels      = ALLOWED_INDELS;
int simd_allowed_diags  = ALLOWED_INDELS;
int delta_m;

#define DISPLAY_OPTION(option, info) {printf("\t\033[37;1m-%s\033[0m: ", option); printf info ; printf("\n");}

#define INTEGER_VALUE_FROM_OPTION(target) {                                     \
    if (sscanf(optarg, "%d", &target) != 1) {                                   \
      ERROR__(("Mandatory numeric parameter missing for option %c.", option));  \
      exit(-1);                                                                 \
    }                                                                           \
  }

#define STRING_VALUE_FROM_OPTION(target) {                                                                            \
    if ((strlen(optarg) == 2) && (optarg[0] == '-') && (optarg[1] != ':') && (index(optstring, optarg[1]) != NULL)) { \
      ERROR__(("Mandatory parameter missing for option %c.", option));                                                \
      exit(-1);                                                                                                       \
    } else {                                                                                                          \
      target = optarg;                                                                                                \
    }                                                                                                                 \
  }

#define HANDLE_INVALID_NUMERIC_VALUE(value, default_) {                                              \
    WARNING__(("Invalid value %d for option %c. Resetting default (%d).", value, option, default_)); \
    value = default_;                                                                                \
  }

#define HANDLE_INVALID_NUMERIC_VALUE_FATAL(value, expected) {                            \
    ERROR__(("Invalid value %d for option %c. Expected: %s.", value, option, expected)); \
    exit(-1);                                                                            \
  }


static void show_usage(char * progname, char* default_seeds) {
  printf("\n%s (beta v%s)\n\n", PROGRAM_NAME, PROGRAM_VERSION);
  printf("\nUSAGE: %s parameters\n\n",progname);
  printf("PARAMETERS:\n");
  printf("\033[37;1mMandatory:\033[0m\n");
  DISPLAY_OPTION("g <genome_file>", ("Name of the reference genome file (.fasta/.multifasta).\n"));
  DISPLAY_OPTION("r <reads_file>", ("Name of the reads file (.csfasta/.fastq/.multifasta).\n"));
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
   * printf("or\n");
   * DISPLAY_OPTION("f <color sequence>", ("Reference sequence.  To be used with  -a and\n"
   *                "\t   optionally -q,  when just the alignment of two  small sequences\n"
   *                "\t   is wanted, instead of -r and -g.\n"));
   * DISPLAY_OPTION("a <color sequence>", ("Query sequence. To be used with -f and optionally\n"
   *                "\t   -p,  when just  the alignment of two  small sequences  is wanted,\n"
   *                "\t   instead of -r and -g.\n"));
   */
  printf("\n\t\033[37;1mWARNING\033[0m: Default is \"--strata like\" mode, otherwise use -M <number>\n\n");
  printf("\n\033[37;1mOptional:\033[0m\n");
  DISPLAY_OPTION("q <qualities_file>", ("Name of the qualities file (qual).\n"));
  DISPLAY_OPTION("s <seed list>", ("The seed family to be used for filtering.  The list should \n"
         "\t   be of the form  \"#-#-#;##--#\" or \"10101;11001\" (seeds separated by ';').\n"
         "\t   These seeds will be applied on every position of the read.  To restrict \n"
         "\t   to certain positions, the  list of wanted positions w.r.t. the read can \n"
         "\t   be mentioned after each seed, as follows: \n"
         "\t   \"#-#-#:0 3 5;##--#:1 2 4 6\". \n"
         "\t   Default seeds:  %s.\n", default_seeds));
  DISPLAY_OPTION("S <seed file>", ("The file containing the seeds to be used for filtering, in \n"
                 "\t   a format similar to the one described for the 's' parameter. Optionally, \n"
                 "\t   for more clarity,  the seeds can be separated by '\\n' instead of ';' in \n"
                 "\t   the file. Default seed file: %s\n", DEFAULT_SEED_FILE));
  DISPLAY_OPTION("l <number>", ("If this option is present,  very frequent k-mers are  removed \n"
                 "\t   from the index.  The numeric parameter establishes which k-mers will be \n"
                 "\t   erased: it expresses the acceptable distance from the average number of \n"
                 "\t   occurrences, measured in number of standard deviations.  For example, a \n"
                                "\t   good value for this parameter is %d. By default, no k-mers are erased. \n", DEFAULT_ACCEPTED_STDEV_DISTANCE));
  DISPLAY_OPTION("j", ("Ignore lowercase symbols from the reference sequences.\n"));
  /*
   * Hiding options that are very unlikely to be used in practice
   */
  /*
   * DISPLAY_OPTION("p <integer sequence>", ("Read quality sequence. To be used with -a and \n"
   *                "\t   -f, when just the alignment of two small sequences is wanted, \n"
   *                "\t    instead of -r and -g.\n"));
   */
  DISPLAY_OPTION("F", ("Only align in forward sense.                      Default: both senses. \n"));
  DISPLAY_OPTION("R", ("Only align in reverse complementary sense.        Default: both senses. \n"));
  DISPLAY_OPTION("o <output_file>", ("The name of the output file  where the mapping result is\n"
                 "\t   written, in SAM format. Default: stdout.\n"));
  DISPLAY_OPTION("m <number>", ("Match score. Default: %d.\n", SCALE_SCORE(MATCH, MAX_READ_QUALITY_LEVEL)));
  DISPLAY_OPTION("x <number>", ("Mismatch penalty. Default: %d.\n"
                            "\n\t\033[37;1mWARNING\033[0m: Match and mismatch scores are \033[37;1mscaled\033[0m according to read qualities, \n"
                        "\tif such qualities are available.\n", SCALE_SCORE(MISMATCH, MAX_READ_QUALITY_LEVEL)));
  DISPLAY_OPTION("d <number>", ("Gap open penalty. Default: %d.\n", GAP_OPEN));
  DISPLAY_OPTION("e <number>", ("Gap extension penalty. Default: %d.\n", GAP_EXTEND));
  DISPLAY_OPTION("i <number>", ("Maximum number of indels in the alignment. Default: %d.\n", ALLOWED_INDELS));
  DISPLAY_OPTION("t <number>", ("Minimum accepted score, above which an alignment is\n"
                                "\t   considered significant. Default: %d.\n", MIN_ACCEPTED_SCORE));
  DISPLAY_OPTION("z <number>", ("The minimum score accepted by the SIMD filter,  above which an \n"
                                "\t   alignment is considered significant. Default: %d.\n", MIN_ACCEPTED_SCORE_SIMD_FILTER));
  DISPLAY_OPTION("u <number>", ("The minimum difference between the first and second \n"
                        "\t   alignment of a given read to consider it unique; Default: disabled.\n"));
  DISPLAY_OPTION("G", ("Perform an \"ordered greedy\" mapping. By default, multiple alignments of\n"
                   "\t   overlapping reads are evaluated at the mapping stage in order to establish\n"
                   "\t   the most relevant candidate mapping position for each read. When this option\n"
                   "\t   is present, only the score of the read's alignment to the reference genome\n"
                   "\t   will be taken into account for choosing the right mapping position.\n"));
  DISPLAY_OPTION("M <number>", ("Perform an \"unordered\" mapping. By default, or in greedy mode, only\n"
                   "\t   one read is kept per reference position.  This option enables any read to\n"
                   "\t   be mapped at any <number> (1..%d) positions per read without considering\n"
                        "\t   this read is the best one for the reference.\n"
                   "\n\t\033[37;1mWARNING\033[0m: Not all mappable reads \033[37;1mwill be mapped\033[0m without this option set to >= 1.\n"
                  , MAP_DETAIL_SIZE));


  DISPLAY_OPTION("v <number>", ("Verbose - output message detail level. %d: no message; \n"
                        "\t   %d: moderate; %d: high, %d: exaggerate. Default: %d.\n",
                        VERBOSITY_NONE, VERBOSITY_MODERATE, VERBOSITY_HIGH, VERBOSITY_ANNOYING, VERBOSITY_DEFAULT));
  DISPLAY_OPTION("h", ("Displays usage and exists.\n"));
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
  char *optstring = "g:r:q:s:S:a:f:p:m:x:d:e:t:z:T:i:v:o:O:l:u:M:jFRhcPHUG";
  /* Input files */
  char *genome_file = NULL, *reads_file = NULL, *qual_file = NULL;
  char *read = NULL, *reference=NULL, *reference_masked=NULL, *quality=NULL;
  char *seeds, *tmp;
  /* Alignment parameters, initialized with the default values from alignment.h */
  ScoreType match = MATCH, mismatch = MISMATCH, gap_open = GAP_OPEN, gap_extend = GAP_EXTEND;

  /* Output files*/
  FILE* output = stdout;

  /* Process command line ... */
  int option;

  cmd_line = argv;
  cmd_line_len = argc;

  seeds = seed__string_from_file(DEFAULT_SEED_FILE);
  if (!seeds) {
    seeds = DEFAULT_SEEDS;
  }

  while((option  = getopt(argc, argv, optstring)) != -1) {
    switch (option) {
    case 'g':
      if (genome_file != NULL) {
        ERROR__(("Genome file previously set with \"%s\".\nUnable to set it again.\n", genome_file));
        exit(-1);
      }
      STRING_VALUE_FROM_OPTION(genome_file);
      break;
    case 'r':
      if (reads_file != NULL) {
        ERROR__(("Read file previously set with \"%s\".\nUnable to set it again.\n", reads_file));
        exit(-1);
      }
      STRING_VALUE_FROM_OPTION(reads_file);
      break;
    case 'q':
      if (qual_file != NULL) {
        ERROR__(("Quality file previously set with \"%s\".\nUnable to set it again.\n", qual_file));
        exit(-1);
      }
      STRING_VALUE_FROM_OPTION(qual_file);
      break;
    case 'S':
      if ((tmp = seed__string_from_file(optarg)) == NULL) {
        char confirmation;
        ERROR__(("Unable to read seeds input file %s.", optarg));
        do {
          printf("Continue (c) with default seeds %s, or abort (a)?\n", seeds);
        } while(scanf("%c", &confirmation) && confirmation != 'a' && confirmation != 'A' && confirmation != 'c' && confirmation != 'C');
        if (confirmation == 'a' || confirmation == 'A') {
          exit(-1);
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
    case 'i':
      INTEGER_VALUE_FROM_OPTION(allowed_indels);
      if (allowed_indels < 0) {
        HANDLE_INVALID_NUMERIC_VALUE_FATAL(allowed_indels, "a positive value or zero");
      }
      if (allowed_indels > INDEL_COUNT_LIMIT) {
        WARNING__(("%s may not handle more than %d indels in the SIMD code (%d diagonals).", PROGRAM_NAME, INDEL_COUNT_LIMIT, INDEL_DATA_VECTOR_SIZE));
      simd_allowed_diags = INDEL_COUNT_LIMIT;
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
          ERROR__(("Could not open or create output file %s.\n", optarg));
          do {
            printf("Continue (c) and output to stdout, or abort (a)?\n");
          } while(scanf("%c", &confirmation) && confirmation != 'a' && confirmation != 'A' && confirmation != 'c' && confirmation != 'C');
          if (confirmation == 'c' || confirmation == 'C') {
            output = stdout;
          } else {
            exit(-1);
          }
        }
        break;
      }
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
    case 'U':
      LIST_UNMAPPED_READS = 1;
      break;
    case 'l':
      FILTER_REPEATS = 1;
      INTEGER_VALUE_FROM_OPTION(ACCEPTED_STDEV_DISTANCE);
      if (ACCEPTED_STDEV_DISTANCE <= 0) {
        HANDLE_INVALID_NUMERIC_VALUE_FATAL(ACCEPTED_STDEV_DISTANCE, "a strictly positive integer");
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

#ifdef _OPENMP
  MAX_THREADS = omp_get_num_procs();
#endif

  delta_m = match - mismatch;

  if (read && reference) {
    printf("\nTwo sequence alignment:\n%s\n%s\n", read, reference);
    if (quality) {
      printf("%s\n", quality);
    }
    single_alignment(read, reference, reference_masked, quality, match, mismatch, gap_open, gap_extend, allowed_indels, simd_allowed_diags, output);
  } else if (reads_file) {
    if (genome_file) {
      printf("\n\nAligning reads against reference.\nInput files: %s, %s, %s.\n\n", genome_file, reads_file, qual_file?qual_file:"(null)");
      reads_against_references(reads_file, qual_file, genome_file, seeds, match, mismatch, gap_open, gap_extend, allowed_indels, simd_allowed_diags, output);
    } else {
      ERROR__(("\nNot enough mandatory parameters provided.\n"));
      show_usage(argv[0], seeds);
    }
  } else {
    ERROR__(("\nNot enough mandatory parameters provided.\n"));
    show_usage(argv[0], seeds);
  }

  if (output && output != stdout) {
    fclose(output);
  }
  return 0;
}


// a test message for svn

// another test change