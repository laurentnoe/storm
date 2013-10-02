#include "reference_data.h"
#include <ctype.h>
#include <unistd.h>
#include <errno.h>

extern int allowed_indels;

int IGNORE_LOWERCASE = 0;

/**
 * Load reference sequence
 */

static int load_reference_sequence(FILE* ref_in, int readlen, ReferenceDBType* db) {
  char line[MAX_LINE_LEN];
  long i_start = ftell(ref_in);

  db->name = NULL;
  VERB_FILTER(VERBOSITY_ANNOYING, INFO__(("pre-loading reference sequence chunk name ...")));
  /* 1) read first line */
  if (!fgets(line, MAX_LINE_LEN, ref_in)) {
    ERROR__(("Empty first dataline in the reference sequence.\n"));
    return RETURN_INPUT_ERR;
  }

  /* 2) fasta or not fasta ... */
  if (get_input_line_type(line) == LINE_TYPE_SEQ_HEADER) {
    /* 2.1) if fasta header skip (but keep the name of) the fasta header... */
    int name_len = strcspn(line + 1, " \r\n");
    if (name_len > 0) {
      SAFE_FAILURE__ALLOC(db->name, (name_len + 1), char);
      strncpy(db->name, line + 1, name_len);
      if (db->name[0] == '=' || db->name[0] == '*') {
        /* Not allowed as first char in SN : arbitrary replace with '-' */
        db->name[0] = '-';
      }
      db->name[name_len] = '\0';
      VERB_FILTER(VERBOSITY_ANNOYING, INFO__(("fasta chunk name found : \"%s\"",db->name)));
      i_start = ftell(ref_in);
    } else {
      ERROR__(("Empty fasta header line in the reference sequence.\n"));
      return RETURN_INPUT_ERR;
    }
  } else {
    /* 2.2) if full ascii sequence, come back to the beginning of it */
    WARNING__(("No fasta header line in the reference sequence.\n"));
    fseek(ref_in, i_start, SEEK_SET);
  }

  VERB_FILTER(VERBOSITY_ANNOYING, INFO__(("pre-loading reference sequence chunk ...")));

  /* 3) read nucleotide data to measure its size */
  int c;
  long size = 0;
  while ((c = (IGNORE_LOWERCASE ? fgetc(ref_in) : toupper(fgetc(ref_in)))) != EOF) {
    if (isspace(c)) {
      continue;
    }
    if (c == SEQ_HEADER_MARKER) {
      /* started to read the next sequence; getting out */
      fseek(ref_in, -1, SEEK_CUR);
      break;
    }
    size++;
  }

  if (size == 0) {
    ERROR__(("Empty fasta chunk line in the reference sequence.\n"));
    return RETURN_INPUT_ERR;
  }

  VERB_FILTER(VERBOSITY_ANNOYING, INFO__(("estimated chunk size : %ld, allocating ...",size)));

  /* 3.2) Allocate the estimated size (>= true size) for sequence and sequence_masked */
  size_t size_shift     = COMPRESSED_LEN_N_BYTES(readlen+allowed_indels,N_BYTES);
  size_t size_allocated = 2*size_shift + COMPRESSED_LEN_N_BYTES(size,N_BYTES);

  SAFE_FAILURE__ALLOC(db->sequence_alloc, size_allocated, CODE_TYPE);
  memset(db->sequence_alloc, '\0', size_allocated);
  db->sequence = db->sequence_alloc + size_shift;

  SAFE_FAILURE__ALLOC(db->sequence_masked_alloc, size_allocated, CODE_TYPE);
  memset(db->sequence_masked_alloc, '\0', size_allocated);
  /* begin masked */
  memset(db->sequence_masked_alloc, '\x55', size_shift);
  /* (x) ... end will be masked once the precise size is known */
  db->sequence_masked = db->sequence_masked_alloc + size_shift;

  VERB_FILTER(VERBOSITY_ANNOYING, INFO__(("loading reference sequence chunk ...")));

  /* 3.3) Go back to first base and load the nucleotides sequence */
  fseek(ref_in, i_start, SEEK_SET);

  long i = 0;
#ifndef NUCLEOTIDES
  CODE_TYPE code_prev = 0;
#endif

  long masked_region_start = -1;
  MaskedRegionType* crt_masked_region = db->masked_regions;

  while ((c = (IGNORE_LOWERCASE ? fgetc(ref_in) : toupper(fgetc(ref_in)))) != EOF) {
    CODE_TYPE code = 0;

    if (isspace(c)) {
      continue;
    }

    if (c == SEQ_HEADER_MARKER) {
      /* started to read the next sequence; getting out */
      fseek(ref_in, -1, SEEK_CUR);
      break;
    }

    if (!IS_VALID_BASE_SYMBOL(c)) {
#ifndef NUCLEOTIDES
      code = code_prev; /* masked regions, eg 'NNNNNNNNN' cannot be coded; instead, GNNNNNNNN is transformed in GGGGGGGGG */
#else
      code = rand()&0x3;  /* random to avoid too much false alignments in the SIMD filter ... */
#endif
      /* does a masked region start here ? */
      if (masked_region_start == -1) {
        masked_region_start = i;
      }
      TO_NTH_CODE(db->sequence_masked, i, 0x1);
    } else {

      code = BASE_SYMBOL_TO_CODE(c);

      /* does a masked region ends here ? */
      if (masked_region_start != -1) {
        if (crt_masked_region == NULL) {
          /* this is the first one */
          SAFE_FAILURE__ALLOC(crt_masked_region, 1, MaskedRegionType);
          db->masked_regions = crt_masked_region;
        } else {
          SAFE_FAILURE__ALLOC(crt_masked_region->next, 1, MaskedRegionType);
          crt_masked_region = crt_masked_region->next;
        }
        crt_masked_region->next  = NULL;
        crt_masked_region->start = masked_region_start;
        crt_masked_region->end   = i;
        VERB_FILTER(VERBOSITY_HIGH, WARNING__(("(%9d-%9d) region masked for reference sequence chunk \"%s\"", crt_masked_region->start+1, crt_masked_region->end, db->name?db->name:"(null)")););
        masked_region_start      = -1;
      }
    }

#ifdef NUCLEOTIDES
    TO_NTH_CODE(db->sequence, i, code);
    i++;
#else
    if (db->first_base == 0xff) {
      code_prev = db->first_base = code;
    } else {
      TO_NTH_CODE(db->sequence, i, COLOR(code_prev, code));
      i++;
      code_prev = code;
    }
#endif
  }

  /* does a masked region ends here ? */
  if (masked_region_start != -1) {
    if (crt_masked_region == NULL) {
      /* this is the first one */
      SAFE_FAILURE__ALLOC(crt_masked_region, 1, MaskedRegionType);
      db->masked_regions = crt_masked_region;
    } else {
      SAFE_FAILURE__ALLOC(crt_masked_region->next, 1, MaskedRegionType);
      crt_masked_region = crt_masked_region->next;
    }
    crt_masked_region->next  = NULL;
    crt_masked_region->start = masked_region_start;
    crt_masked_region->end   = i;
    VERB_FILTER(VERBOSITY_HIGH, WARNING__(("(%9d-%9d) region masked for reference sequence chunk \"%s\"", crt_masked_region->start+1, crt_masked_region->end, db->name?db->name:"(null)")););
    masked_region_start      = -1;
  }

  db->size = i;

  VERB_FILTER(VERBOSITY_ANNOYING, INFO__(("true chunk size : %d",db->size)));

  /* (x) end mask here with the last byte, then size_shift */
  while (COMPRESSED_OFFSET(i)) {
    TO_NTH_CODE(db->sequence_masked, i, 0x1);
    i++;
  }
  memset(db->sequence_masked+COMPRESSED_IDX(i), '\x55', size_shift);
  return db->size;
}

/**
 * Load reference sequence
 */
int load_reference_db(const char* reference_filename, int readlen, ReferenceDBType** db) {
  char line[MAX_LINE_LEN];
  FILE* ref_in = fopen(reference_filename, "r");
  int i=0;
  int sequence_count = 0;
  long long int full_size = 0;
  if (!ref_in) {
    ERROR__(("Failed to read input file %s (reference sequence).\n", reference_filename));
    exit(RETURN_INPUT_ERR);
  }
  // browse the file once
  /* how many sequences are there?  */
  /* Expected format: multifasta (one 'header' line, followed by the sequence) */
  // So count headers...
  VERB_FILTER(VERBOSITY_ANNOYING, INFO__(("pre-loading reference sequence \"%s\"", reference_filename)));
  while (fgets(line, MAX_LINE_LEN, ref_in)) {
    if (get_input_line_type(line) == LINE_TYPE_SEQ_HEADER) {
      ++sequence_count;
    }
  }
  VERB_FILTER(VERBOSITY_ANNOYING, INFO__(("reference sequence \"%s\" : #%d chunks", reference_filename, sequence_count)));

  if (sequence_count == 0) {
    /* no header in the reference ... lets try without */
    sequence_count = 1;
  }

  /* Allocate */
  SAFE_FAILURE__ALLOC(*db, sequence_count, ReferenceDBType);

  /* Go back to the beginning and start loading */
  fseek (ref_in, 0, SEEK_SET);

  for (i = 0; i < sequence_count; ++i) {
    int size;
    (*db)[i].masked_regions  = NULL;
    (*db)[i].name            = NULL;
    (*db)[i].size            = 0;
    (*db)[i].sequence_alloc         = NULL;
    (*db)[i].sequence               = NULL;
    (*db)[i].sequence_masked_alloc  = NULL;
    (*db)[i].sequence_masked        = NULL;
#ifndef NUCLEOTIDES
    (*db)[i].first_base     = 0xff;
#endif
    VERB_FILTER(VERBOSITY_ANNOYING, INFO__(("reference sequence \"%s\" : loading chunk %d/%d ...", reference_filename, i+1,sequence_count)));
    if ((size = load_reference_sequence(ref_in, readlen, &((*db)[i]))) <= 0) {
      /* something strange happened, stop loading and return what we have */
      sequence_count = i;
      WARNING__(("problem when loading chunk %d for reference sequence \"%s\"", i+1, reference_filename));
      break;
    }
    full_size += size;
  }
  fclose(ref_in);
  VERB_FILTER(VERBOSITY_HIGH, INFO__(("reference sequence \"%s\" : #%d chunks (full size:%lld)", reference_filename, sequence_count, full_size)));
  return sequence_count;
}

/**
 * Destroys a reference database
 */
void clear_reference_db(ReferenceDBType* db) {
  while (db->masked_regions != NULL) {
    MaskedRegionType* tmp = db->masked_regions;
    db->masked_regions = db->masked_regions->next;
    free(tmp);
  }
  if (db->name) {
    free(db->name);
    db->name = NULL;
  }
  free(db->sequence_alloc);
  db->sequence        = db->sequence_alloc        = NULL;
  free(db->sequence_masked_alloc);
  db->sequence_masked = db->sequence_masked_alloc = NULL;
}

