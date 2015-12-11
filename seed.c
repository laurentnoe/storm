#include "seed.h"
#include "index.h"
#include "load_data.h"
#include "util.h"


#define IS_MATCH(symbol) ((symbol) == MATCH_SYMBOL || (symbol) == MATCH_CODE)
#define GET_SEED_MASK(symbol) ((symbol) == MATCH_SYMBOL || (symbol) == MATCH_CODE) ? SEED_MATCH_MASK : (((symbol) == WILDCARD_SYMBOL || (symbol) == WILDCARD_CODE) ? SEED_WILDCARD_MASK : (SEED_UNDEFINED))
#define IS_UNDEFINED(code) (!((code) ^ SEED_UNDEFINED))
#define IS_SEED_MATCH(code) ((code) & SEED_MATCH_MASK)



/* the Windows world company ... */
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
#define strtok_r strtok_s
#endif


/**
 * Create a seed from a pattern
 * @param pattern Accepted forms: "1        001...:pos1 pos2 pos3..." or "#-##--#..:pos1 pos2 pos3..."
 * @return The address of the created seed.
 */
SeedType* seed__create(const char* pattern) {
  int i;
  char* pos_list, *scanner, *saveptr, *token;
  unsigned long long int mask = 1;
  SeedType* seed;
  SAFE_FAILURE__ALLOC(seed, 1, SeedType);
  //process the pattern before ':'
  pos_list = strchr(pattern, SEED_POS_LIST_DELIMITER);
  seed->length = strlen(pattern) - ((pos_list) ? strlen(pos_list) : 0);
  SAFE_FAILURE__ALLOC(seed->seed, seed->length, CODE_TYPE);
  seed->weight = 0;
  seed->mask = 0;
  seed->positions = NULL;
  seed->positions_count = 0;
  for (i = 0; i < seed->length; ++i) {
    seed->seed[i] = GET_SEED_MASK(pattern[i]);
    if (IS_UNDEFINED(seed->seed[i])) {
      seed__destroy(seed);
      return NULL;
    }
    if (IS_SEED_MATCH(seed->seed[i])) {
      seed->weight++;
      seed->mask |= mask;
    }
    mask <<= 1;
  }
  if (seed->weight == 0) {
    seed__destroy(seed);
    return NULL;
  }

  //process the pos_list
  if (pos_list && strlen(pos_list) > 1) {
    char *pos_ptr;
    scanner = pos_list + 1; // skip the separator

    /* Number of elements (if any) = number of delimiters + 1 */
    if (sscanf(scanner, "%d", &i) == 1) {
      seed->positions_count = 1;

      /* count the positions */
      while (scanner && (
        (pos_ptr = strchr(scanner, SEED_POS_DELIMITER_1))
        ||
        (pos_ptr = strchr(scanner, SEED_POS_DELIMITER_2))
        )) {
        seed->positions_count++;
        scanner = pos_ptr + 1;
      }
      /* create the list */
      scanner = pos_list + 1;
      SAFE_FAILURE__ALLOC(seed->positions, seed->positions_count, int);

      for (i = 0, pos_ptr = scanner; ; i++, pos_ptr = NULL) {
        token = strtok_r(pos_ptr, SEED_POS_DELIMITER, &saveptr);
        if (token == NULL || sscanf(token,"%d", &(seed->positions[i])) != 1) {
          seed->positions_count = i;
          break;
        }
      } /* FOR: parse position list*/
    } /* IF: There is a number in the list */
  } /* IF: the position list is not an empty string */
  return seed;
}

/**
 * Creates an array of seeds from an input string of the form
 * <seed_pattern>(<seed_delimiter><seed_pattern>)*
 * @param src The input string
 * @param seeds Adress where the resulted array is put
 * @return The number of created seeds
 */
int seed__parse_list(const char* src, SeedType*** seeds) {
  char *token, *delim = ";", *src_copy, *src_copy2, *ssaveptr;
  int seed_count = 1, j;

  *seeds = NULL;

  if (strlen(src) == 0) {
    return 0;
  }
  SAFE_FAILURE__ALLOC(src_copy, strlen(src) + 1, char);
  strcpy(src_copy, src);

  src_copy2 = src_copy;

  while ((src_copy2 = strchr(src_copy2, SEED_DELIMITER))) {
    ++seed_count;
    ++src_copy2;
  }

  SAFE_FAILURE__ALLOC(*seeds, seed_count, SeedType*);

  for (j = 0, src_copy2 = src_copy; ; j++, src_copy2 = NULL) {
    token = strtok_r(src_copy2, delim, &ssaveptr);
    if (token == NULL) {
      seed_count = j;
      break;
    }
    (*seeds)[j] = seed__create(token);
    if (! (*seeds)[j]) {
      for (seed_count = 0; seed_count < j; ++seed_count) {
        seed__destroy((*seeds)[seed_count]);
      }
      free(*seeds);
      return RETURN_INPUT_ERR;
    }
  }
  free(src_copy);
  return seed_count;
}

/**
 * Read a seed list from a file
 */
char* seed__string_from_file(const char* filename) {
  char buffer[MAX_LINE_LEN];
  char *string;
  int size = 0;
  FILE* file = fopen(filename, "r");
  if (!file) {
    return NULL;
  }
  while (fgets(buffer, MAX_LINE_LEN, file)) {
    size += strlen(buffer)+1;
  }

  SAFE_FAILURE__ALLOC(string, size+1, char);
  memset(string,'\0',size+1);

  fseek(file, 0, SEEK_SET);

  size = 0;
  while (fgets(string + size, MAX_LINE_LEN, file)) {
    if (string[size] == '%') {
      /* ignore commented line */
      continue;
    }
    size = strlen(string);
    /* remove trailing '\n' and spaces */
    while (size > 0 && (string[size - 1] == '\n' || string[size - 1] == '\r' || string[size - 1] == ' ' || string[size - 1] == '\t')) {
      string[size - 1] = '\0';
      size--;
    }
    /* add delimiter if none */
    if (string[size - 1] != SEED_DELIMITER) {
      size++;
      string[size - 1] = SEED_DELIMITER;
      string[size]     = '\0';
    }
  }
  fclose(file);

  return string;
}

/**
 * Apply a seed to the given sequence
 * @param sequence
 * @return The code of the indexed zone
 */
int seed__apply(const SeedType *seed, const CODE_TYPE* sequence, const int pos) {
  int key = 0, i;
  for (i = 0; i < seed->length; ++i) {
    if (IS_SEED_MATCH(seed->seed[i])) {
      key <<= SEED_MASK_SIZE;
      key |= SEED_MATCH_MASK & sequence[pos + i];
    }
  }
  return key;
}

/**
 * Apply a seed to the given compressed sequence
 * @param sequence
 * @param offset
 * @return The code of the indexed zone
 */
int seed__apply_to_compressed(const SeedType *seed, const CODE_TYPE* sequence, const int pos) {
  int key = 0;
  int i;
  for (i = 0; i < seed->length; ++i) {
    if (IS_SEED_MATCH(seed->seed[i])) {
      key <<= SEED_MASK_SIZE;
      key |= SEED_MATCH_MASK & NTH_CODE(sequence, (pos + i));
    }
  }
  return key;
}

/**
 * Generate a friendly representation of the seed
 * @param seed
 * @return A string holding the representation
 */
char* seed__to_string(const SeedType *seed) {
  int i;
  char* seed_str;
  SAFE_FAILURE__ALLOC(seed_str, seed->length + 1, char);
  memset(seed_str, '\0', seed->length + 1);
  for (i = 0; i < seed->length; ++i) {
    if (IS_SEED_MATCH(seed->seed[i])) {
      seed_str[i] = MATCH_SYMBOL;
    } else {
      seed_str[i] = WILDCARD_SYMBOL;
    }
  }
  return seed_str;
}
/**
 * Display a seed
 * @param seed
 */
void seed__display(const SeedType *seed) {
  int i;
  for (i = 0; i < seed->length; ++i) {
    if (IS_SEED_MATCH(seed->seed[i])) {
      MESSAGE__("%c", MATCH_SYMBOL);
    } else {
      MESSAGE__("%c", WILDCARD_SYMBOL);
    }
  }
  if (seed->positions) {
    MESSAGE__(":");
    for (i = 0; i < seed->positions_count; ++i) {
      MESSAGE__("%d ", seed->positions[i]);
    }
  }
  MESSAGE__("\n");
}

/**
 * Destroy a seed
 * @param seed
 */
void seed__destroy(SeedType* seed) {
  free(seed->seed);
}
