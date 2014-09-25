#include <unistd.h>
#include <string.h>
#include "reads_data.h"


/**
 * Reverse-complement a read
 */
void read__reverse(const ReadDataType* src, ReadDataType* dest, int len) {
  /* WARNING: only sequence and quality fields are cloned;
   * also, the dest is considered to have all the allocated space needed
   * by default, the read is compressed
   */
#ifndef NUCLEOTIDES
  dest->first_base = BASE_CODE_COMPLEMENTARY(base_compressed_color_transformation(src->first_base, src->sequence, len, 0));
#endif
  sequence__reverse_compressed(src->sequence, dest->sequence, len);
  if (src->quality) {
    if (!dest->quality) {
      SAFE_FAILURE__ALLOC(dest->quality, COMPRESSED_Q_LEN(len), QUAL_TYPE);
    }

#ifndef NUCLEOTIDES
    quality__reverse_compressed(src->first_qual, src->quality, dest->quality, len);
    dest->first_qual = NTH_QUAL(src->quality, len-1);
#else
    quality__reverse_compressed(src->quality, dest->quality, len);
#endif
  } else {
    if (dest->quality) {
      free (dest->quality);
    }
    dest->quality = NULL;
  }
}


/**
 * Sort reads function (alphanumeric order)
 */
static int gv_read_len = 0;

static int sort_reads_fun(const void *a, const void *b) {
  ReadDataType * r_a = (ReadDataType *)a;
  ReadDataType * r_b = (ReadDataType *)b;

  /* read "a" tag : not clean to do this here ... but must be parallel if psort ... and very few are computed as polyA "0" */
  if (!r_a->tag) {
    int i;
    unsigned long c_tag = 0;
    for (i = 0; i < gv_read_len; i++) {
      c_tag <<= 2 ; c_tag |= ((NTH_CODE(r_a->sequence, i) + 1) & 0x3) ^ 0x1; /* (x):[acgt] order -> ((x)+1)&0x3:[tacg] order -> (((x)+1)&0x3)^0x1:[atgc] order : so more weigth given to "gc" more than "at" */
      r_a->tag = MAX(r_a->tag,c_tag);
    }
  }
  /* read "b" tag : not clean to do this here ... but must be parallel if psort ... and very few are computed as polyA "0" */
  if (!r_b->tag) {
    int i;
    unsigned long c_tag = 0;
    for (i = 0; i < gv_read_len; i++) {
      c_tag <<= 2 ; c_tag |= ((NTH_CODE(r_b->sequence, i) + 1) & 0x3) ^ 0x1;
      r_b->tag = MAX(r_b->tag,c_tag);
    }
  }

  /* comparison of tags */
  if (r_a->tag < r_b->tag)
    return -1;
  if (r_a->tag > r_b->tag)
    return +1;
  return 0;
}


void psort() __attribute__((weak));

/**
 * Sort a read database (alphanumeric order)
 */
int sort_reads_db(ReadsDBType * db) {
  gv_read_len = db->read_len;

  if (psort)
    psort((void *)db->reads/*base*/,
          (size_t)db->size /*nel*/,
          (size_t)(sizeof(ReadDataType))/*width*/,
          sort_reads_fun/*int (*compar)(const void *, const void *)*/);
  else
    qsort((void *)db->reads/*base*/,
          (size_t)db->size /*nel*/,
          (size_t)(sizeof(ReadDataType))/*width*/,
          sort_reads_fun/*int (*compar)(const void *, const void *)*/);
  return 0;
}


/**
 * Load fasta/csfasta reads file
 */
int load_reads_db_fasta_csfasta(const char* reads_filename, const char* quals_filename, ReadsDBType* db) {
  char line[MAX_LINE_LEN];
  char* read_result;
  FILE* reads_in = NULL;
  FILE* quals_in = NULL;
  short* quality;

  if (reads_filename)
    reads_in = fopen(reads_filename, "r");

  if (!reads_in) {
    ERROR__("Failed to read input file %s (read sequences).", reads_filename);
    exit(RETURN_INPUT_ERR);
  }

  if (quals_filename) {
    quals_in = fopen(quals_filename, "r");
    if (!quals_in) {
      ERROR__("Failed to read input file %s (quality of read sequences).\n", quals_filename);
      exit(RETURN_INPUT_ERR);
    }
  }

  db->size = 0;
  db->read_len = 0;
  db->name = NULL;

  /* First scan: count how many reads are there in the file */
  VERB_FILTER(VERBOSITY_ANNOYING, INFO__("pre-loading reads from \"%s\" ...",reads_filename));

  /* Skip comments at the beginning; after this, i will point to the start of the first non-comment line */
  long i_start = 0;
  do {
    i_start = ftell(reads_in);
    read_result = fgets(line, MAX_LINE_LEN, reads_in);
    /* Is there a title in the comment ? */
    if (read_result && !(db->name) && strncmp(line, TITLE_LINE_MARKER, TITLE_LINE_MARKER_LEN) == 0) {
      int name_len = strcspn(line + TITLE_LINE_MARKER_LEN, "\r\n");
      SAFE_FAILURE__ALLOC(db->name, (name_len + 1), char);
      strncpy(db->name, line + TITLE_LINE_MARKER_LEN, name_len);
      db->name[name_len] = '\0';
    }
  } while (read_result && get_input_line_type(line) == LINE_TYPE_COMMENT);

  if (!read_result) {
    ERROR__("Failed to read input file %s (read sequences) first line.", reads_filename);
    exit(RETURN_INPUT_ERR);
  }

  /* The last line is not a comment, it is the header of the first sequence */
  do {
    if (get_input_line_type(line) == LINE_TYPE_SEQ_HEADER &&
        fgets(line, MAX_LINE_LEN, reads_in) &&
        get_input_line_type(line) == LINE_TYPE_SEQ) {

      if (db->read_len == 0) {

#ifdef NUCLEOTIDES
        db->read_len = strcspn(line, "\r\n");
#else
        db->read_len = strcspn(line, "\r\n") - 2;
        /* without the first 2 chars, which are is a base and the first color,
         * used to obtain the real first base
         */
#endif

        while ((read_result = fgets(line, MAX_LINE_LEN, reads_in)) && get_input_line_type(line) == LINE_TYPE_SEQ)
          db->read_len += strcspn(line, "\r\n");

        if (db->read_len <= 0) {
          ERROR__("\nReads file seams corrupt: the first read's length is %d. Aborting.\n\n", db->read_len);
          exit(RETURN_INPUT_ERR);
        }
      } else {
#ifdef NUCLEOTIDES
        int current_length = strcspn(line, "\r\n");
#else
        int current_length = strcspn(line, "\r\n") - 2;
#endif

        while ((read_result = fgets(line, MAX_LINE_LEN, reads_in)) && get_input_line_type(line) == LINE_TYPE_SEQ)
          current_length += strcspn(line, "\r\n");

        if (current_length != db->read_len) {
          ERROR__("\nThe read #%ld does not have the expected length (length is %d, expected %d).\n"
                   "All the reads must be of equal length for the parallel processing.\n\n",
                   db->size + 1,
                   current_length,
                   db->read_len);
          exit(RETURN_INPUT_ERR);
        }
      }
      ++db->size;
    }
  } while (read_result);

  /* The last line is a read; we can obtain the read length */
  VERB_FILTER(VERBOSITY_ANNOYING, INFO__("%ld reads found from \"%s\" (read length %d), allocating ...",db->size,reads_filename,db->read_len));

  /* Allocate */
  SAFE_FAILURE__ALLOC(db->reads, db->size, ReadDataType);
  memset(db->reads,'\0',db->size * sizeof(ReadDataType));

  /* Second scan: load reads */
  /* Go back to the first non-comment line */
  VERB_FILTER(VERBOSITY_MODERATE, MESSAGE__("Loading reads..."););
  fseek(reads_in, i_start, SEEK_SET);

  {
    char * linebuffer = NULL;
#ifdef NUCLEOTIDES
    SAFE_FAILURE__ALLOC(linebuffer,db->read_len+1,char);
    memset(linebuffer,'\0',db->read_len+1);
#else
    SAFE_FAILURE__ALLOC(linebuffer,db->read_len+3,char);
    memset(linebuffer,'\0',db->read_len+3);
#endif
    long i = 0;
    int line_len = 0;
    while (fgets(line, MAX_LINE_LEN, reads_in)) {
    nl:
      switch (get_input_line_type(line)) {
      case LINE_TYPE_SEQ_HEADER:
        {
          // we suppose that the header is read all at once
          delete_trailing_nl(line);
          // start after the first '>', and cut at the first invalid char (eg '@' or ' ') of the header
          line_len = 1;
          while (line[line_len] > ' ' && line[line_len] <= '~' && line[line_len] != SEQ_HEADER_MARKER_FASTQ){
            line_len++;
            if (line_len >= 256) {
              WARNING__("\nRead #%ld : name too long (\"%s\").\n", i+1, line);
              break;
            }
          }
          line[line_len] = '\0';
          // info size: line_len - 1 (without the '>'); allocate line_len (+ 1 for '\0' but - 1 for '>')
          SAFE_FAILURE__ALLOC(db->reads[i].info, line_len, char);
          strncpy(db->reads[i].info, line + 1 /* without '>' */, line_len - 1);
          db->reads[i].info[line_len - 1] = '\0';
          db->reads[i].quality = NULL;
          break;
        }
      case LINE_TYPE_SEQ:
        {
          char * linebuffer_shift  = linebuffer;
#ifdef NUCLEOTIDES
          int    linebuffer_remain = db->read_len;
#else
          int    linebuffer_remain = db->read_len+2;
#endif
          do {
            delete_trailing_nl(line);
            line_len = strlen(line);
            if (line_len <= linebuffer_remain) {
              strncpy(linebuffer_shift, line, line_len);
              linebuffer_shift  += line_len;
              linebuffer_remain -= line_len;
            } else {
              ERROR__("\nThe read #%ld does not have the expected format.\n",
                       i + 1);
              exit(-1);
            }
          } while ((read_result = fgets(line, MAX_LINE_LEN, reads_in)) && get_input_line_type(line) == LINE_TYPE_SEQ);
          // INFO__("%8d/%8d : \"%s\" strlen:%d\n", i, db->size, linebuffer, (int)strlen(linebuffer));fflush(NULL);
#ifdef NUCLEOTIDES
          db->reads[i].sequence = string_to_compressed_code(linebuffer);
#else
          // also, we suppose that the sequence is read all at once, since it is known to be short
          // the first base in the file is always the same; the real first base is obtained by transforming it with the first color
          db->reads[i].first_base = BASE_SYMBOL_TO_CODE(linebuffer[0]);
          db->reads[i].first_base = TRANSFORM(db->reads[i].first_base, STRIP(linebuffer[1]));
          // copy the color sequence after the fake first base and the first color
          db->reads[i].sequence = string_to_compressed_code(linebuffer+2);
#endif
          db->reads[i].tag = 0;
          i++;
          if (i == db->size) {
            goto eol;
          }
          VERB_FILTER(VERBOSITY_MODERATE, display_progress(i, db->size, 0););
          goto nl;
        }
      default:
        // ignore other kind of lines
        break;
      }
    }
  eol:
    VERB_FILTER(VERBOSITY_MODERATE, display_progress(db->size, db->size, 0););
    fclose(reads_in);
    free(linebuffer);
  }


  /* Try to load qualities (csfasta only, not fasta) */
  if (quals_in) {
    long i;
    VERB_FILTER(VERBOSITY_MODERATE, MESSAGE__("Loading qualities..."););
#ifndef NUCLEOTIDES
    SAFE_FAILURE__ALLOC(quality, db->read_len + 1, short);
    memset(quality, '\0', (db->read_len + 1) * sizeof(short));
#else
    SAFE_FAILURE__ALLOC(quality, db->read_len, short);
    memset(quality, '\0', db->read_len * sizeof(short));
#endif
    /* Load qualities */
    /* For the moment, no rigorous  verification is done, qualities are considered in the right order */
    for (i = 0; i < db->size; ) {
      if(!fgets(line, MAX_LINE_LEN, quals_in)) {
        WARNING__("\nRead #%ld : quality line not readable (\"%s\").\n", i+1, quals_filename);
        break;
      }
      if (get_input_line_type(line) == LINE_TYPE_SEQ) {
#ifndef NUCLEOTIDES
        if (parse_integer_sequence_to(line, quality, db->read_len + 1)) {
          WARNING__("\nRead #%ld : quality line too short (\"%s\").\n", i+1, quals_filename);
        }
        db->reads[i].first_qual = QUALITY_LEVEL(quality[0]);
        db->reads[i].quality = compress_quality_sequence(quality, db->read_len, 1);
#else
        if (parse_integer_sequence_to(line, quality, db->read_len)) {
          WARNING__("\nRead #%ld : quality line too short (\"%s\").\n", i+1, quals_filename);
        }
        db->reads[i].quality = compress_quality_sequence(quality, db->read_len, 0);
#endif
        i++;
      }
      VERB_FILTER(VERBOSITY_MODERATE, display_progress(i, db->size, 0););
    }
    VERB_FILTER(VERBOSITY_MODERATE, display_progress(db->size, db->size, 0););
    fclose(quals_in);
  }

  /* Set a name if none */
  if (db->name == NULL) {
    if (db->size && db->reads[0].info != NULL && db->reads[db->size-1].info != NULL) {
      int l = 0, t = 0;
      while( db->reads[0].info[l] != '\0' && db->reads[db->size-1].info[l] != '\0' &&
             db->reads[0].info[l]         == db->reads[db->size-1].info[l] ) {
        if (
            (
             (t == 0) &&
             ((db->reads[0].info[l] >= '!' && db->reads[0].info[l] <= ')') ||
              (db->reads[0].info[l] >= '+' && db->reads[0].info[l] <= '<') ||
              (db->reads[0].info[l] >= '>' && db->reads[0].info[l] <= '~'))
             ) || (
              (t > 0) &&
              (db->reads[0].info[l] >= '!' && db->reads[0].info[l] <= '~')
             )
           )
        t = l+1;
        l++;
      }
      if (t > 0) {
        SAFE_FAILURE__ALLOC(db->name,t+1,sizeof(char));
        strncpy(db->name, db->reads[0].info, t);
      }
    }
  }
  return db->size;
}


/**
 * Load fastq reads file
 */
int load_reads_db_fastq(const char* reads_filename, ReadsDBType* db) {
  char line[MAX_LINE_LEN];
  char* read_result;
  FILE* reads_in = NULL;
  short* quality;
  int lineNo=-1, offset=-1;

  if (reads_filename)
    reads_in = fopen(reads_filename, "r");

  if (!reads_in) {
    ERROR__("Failed to read input file %s (read sequences).", reads_filename);
    exit(RETURN_INPUT_ERR);
  }

  db->size = 0;
  db->read_len = 0;
  db->name = NULL;

  /* First scan: count how many reads are there in the file */
  VERB_FILTER(VERBOSITY_ANNOYING, INFO__("pre-loading reads from \"%s\" ...",reads_filename));

  /* Skip comments at the beginning; after this, i will point to the start of the first non-comment line */
  long i_start = 0;
  do {
    offset++;
    lineNo++;
    i_start = ftell(reads_in);
    read_result = fgets(line, MAX_LINE_LEN, reads_in);
    /* Is there a title in the comment ? */
    if (read_result && !(db->name) && strncmp(line, TITLE_LINE_MARKER, TITLE_LINE_MARKER_LEN) == 0) {
      int name_len = strcspn(line + TITLE_LINE_MARKER_LEN, "\r\n");
      SAFE_FAILURE__ALLOC(db->name, (name_len + 1), char);
      strncpy(db->name, line + TITLE_LINE_MARKER_LEN, name_len);
      db->name[name_len] = '\0';
    }
  } while (read_result && get_input_line_type_fastq(line, lineNo, offset) == LINE_TYPE_COMMENT);

  if (!read_result) {
    ERROR__("Failed to read input file %s (read sequences) first line.", reads_filename);
    exit(RETURN_INPUT_ERR);
  }

  /* The last line is not a comment, it is the header of the first sequence */
  int lineType, nextLineType=-1;
  do {
    if ((lineType = get_input_line_type_fastq(line, lineNo++, offset)) == LINE_TYPE_SEQ_HEADER &&
        fgets(line, MAX_LINE_LEN, reads_in) &&
        (nextLineType = get_input_line_type_fastq(line, lineNo++, offset)) == LINE_TYPE_SEQ) {
      if (db->read_len == 0) {
#ifdef NUCLEOTIDES
        db->read_len = strcspn(line, "\r\n");
#else
        db->read_len = strcspn(line, "\r\n") - 2;
        /* without the \n and the first 2 chars, which are is a base and the first color, used to obtain the real first base */
#endif
        if (db->read_len <= 0) {
          ERROR__("\nReads file seams corrupt: the first read's length is %d. Aborting.\n\n", db->read_len);
          exit(RETURN_INPUT_ERR);
        }
      } else {
#ifdef NUCLEOTIDES
        if (strcspn(line, "\r\n") != db->read_len)
#else
        if (strcspn(line, "\r\n") - 2 != db->read_len)
#endif
          {
            ERROR__("\nThe read #%ld does not have the expected length (length is %d, expected %d).\n"
                     "All the reads must be of equal length for the parallel processing.\n\n",
                     db->size + 1,
#ifdef NUCLEOTIDES
                     (int)strcspn(line, "\r\n"),
#else
                     (int)strcspn(line, "\r\n") - 2,
#endif

                     db->read_len);
            exit(RETURN_INPUT_ERR);
          }
      }
      ++db->size;
    } else if ((lineType == LINE_TYPE_SEQ_HEADER && nextLineType != LINE_TYPE_SEQ) || lineType == LINE_TYPE_UNKNOWN) {
      ERROR__("\nReads file seams corrupt at line %d. Exiting.\n", lineNo);
      exit(RETURN_INPUT_ERR);
    }
  } while (fgets(line, MAX_LINE_LEN, reads_in));

  /* The last line is a read; we can obtain the read length */
  VERB_FILTER(VERBOSITY_ANNOYING, INFO__("%ld reads found from \"%s\" (read length %d), allocating ...",db->size,reads_filename,db->read_len));

  /* Allocate */
  SAFE_FAILURE__ALLOC(db->reads, db->size, ReadDataType);
  memset(db->reads,'\0',db->size * sizeof(ReadDataType));

  SAFE_FAILURE__ALLOC(quality, db->read_len, short);
  memset(quality,'\0',db->read_len * sizeof(short));

  /* Second scan: load reads */
  /* Go back to the first non-comment line */
  fseek(reads_in, i_start, SEEK_SET);
  VERB_FILTER(VERBOSITY_MODERATE, MESSAGE__("Loading reads..."););

  long i = -1;
  int line_len = 0;
  lineNo = offset;
  while (i < db->size) {
    if (!fgets(line, MAX_LINE_LEN, reads_in)) {
      if (++i == db->size)
        continue;
      return RETURN_INPUT_ERR;
    }
    switch (get_input_line_type_fastq(line, lineNo++, offset)) {
    case LINE_TYPE_SEQ_HEADER:
      {
      ++i;
      // we suppose that the header is read all at once
      delete_trailing_nl(line);
      // start after the first '@', and cut at the first invalid char (eg '@' or ' ') of the header
      line_len = 1;
      while (line[line_len] > ' ' && line[line_len] <= '~' && line[line_len] != '@'){
        line_len++;
        if (line_len >= 256) {
        WARNING__("\nRead #%ld : name too long (\"%s\").\n", i+1, line);
          break;
      }
      }
      line[line_len] = '\0';
      // info size: line_len - 1 (without the '@'); allocate line_len (+ 1 for '\0' but - 1 for '@')
      SAFE_FAILURE__ALLOC(db->reads[i].info, line_len, char);
      strncpy(db->reads[i].info, line + 1 /* without '@' */, line_len - 1);
      db->reads[i].info[line_len - 1] = '\0';
      db->reads[i].quality = NULL;
      break;
      case LINE_TYPE_SEQ:
      delete_trailing_nl(line);
#ifdef NUCLEOTIDES
      db->reads[i].sequence = string_to_compressed_code(line);
#else
      // also, we suppose that the sequence is read all at once, since it is known to be short
      // the first base in the file is always the same; the real first base is obtained by transforming it with the first color
      db->reads[i].first_base = BASE_SYMBOL_TO_CODE(line[0]);
      db->reads[i].first_base = TRANSFORM(db->reads[i].first_base, STRIP(line[1]));
      // copy the color sequence after the fake first base and the first color
      db->reads[i].sequence = string_to_compressed_code(line+2);
#endif
      db->reads[i].tag = 0;
      break;
      }
    case LINE_TYPE_QUAL_HEADER:
      break;
    case LINE_TYPE_QUAL:
      // parse the characher string encoding the quality
#ifndef NUCLEOTIDES
      db->reads[i].first_qual = QUALITY_LEVEL(FASTQ_PARSE_QUALITY_FROM_CHAR(line[0]));
#endif
      delete_trailing_nl(line);
      memset(quality, 0, (db->read_len)*sizeof(short));
      int p;
#ifdef NUCLEOTIDES
      p = 0;
#else
      p = 1;
#endif
      for (; p < MIN(db->read_len, strlen(line)); ++p) {
        quality[p] = FASTQ_PARSE_QUALITY_FROM_CHAR(line[p]);
      }
#ifdef NUCLEOTIDES
      db->reads[i].quality = compress_quality_sequence(quality, db->read_len, 0);
#else
      db->reads[i].quality = compress_quality_sequence(quality, db->read_len, 1);
#endif
      break;
    default:
      // ignore other kind of lines
      break;
    }
    VERB_FILTER(VERBOSITY_MODERATE, display_progress(i, db->size, 0););
  }
  VERB_FILTER(VERBOSITY_MODERATE, display_progress(db->size, db->size, 0););
  fclose(reads_in);

  /* Set a name if none */
  if (db->name == NULL) {
    if (db->size && db->reads[0].info != NULL && db->reads[db->size-1].info != NULL) {
      int l = 0, t = 0;
      while( db->reads[0].info[l] != '\0' && db->reads[db->size-1].info[l] != '\0' &&
             db->reads[0].info[l]         == db->reads[db->size-1].info[l] ) {
        if (
            (
             (t == 0) &&
             ((db->reads[0].info[l] >= '!' && db->reads[0].info[l] <= ')') ||
              (db->reads[0].info[l] >= '+' && db->reads[0].info[l] <= '<') ||
              (db->reads[0].info[l] >= '>' && db->reads[0].info[l] <= '~'))
             ) || (
              (t > 0) &&
              (db->reads[0].info[l] >= '!' && db->reads[0].info[l] <= '~')
             )
           )
        t = l+1;
        l++;
      }
      if (t > 0) {
        SAFE_FAILURE__ALLOC(db->name,t+1,sizeof(char));
        strncpy(db->name, db->reads[0].info, t);
      }
    }
  }
  return db->size;
}

int load_reads_db(const char* reads_filename, const char* quals_filename, ReadsDBType* db) {
  char * pos_dot = rindex(reads_filename, '.');
  if ((pos_dot != NULL)  &&  ((strcmp(pos_dot,".fastq") == 0) || (strcmp(pos_dot,".fq") == 0) || (strcmp(pos_dot,".fastq2") == 0) ||( strcmp(pos_dot,".fq2") == 0))) {
    return load_reads_db_fastq(reads_filename, db);
  } else {
    return load_reads_db_fasta_csfasta(reads_filename, quals_filename, db);
  }
}

/**
 * Destroys a read database
 */
void clear_reads_db(ReadsDBType* db) {
  long i;
  for (i = 0; i < db->size; ++i) {
    free(db->reads[i].info);
    free(db->reads[i].sequence);
    if (db->reads[i].quality) {
      free(db->reads[i].quality);
    }
  }
  free(db->reads);
  if (db->name) {
    free(db->name);
  }
}

