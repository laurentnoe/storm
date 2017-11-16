#ifndef _SEED_H_
#define _SEED_H_

#include "codes.h"

#define SEED_MATCH_MASK    0x3
#define SEED_WILDCARD_MASK 0x0
#define SEED_UNDEFINED     0x1
#define SEED_MASK_SIZE       2

#define MATCH_SYMBOL    '#'
#define MATCH_CODE      '1'
#define WILDCARD_SYMBOL '-'
#define WILDCARD_CODE   '0'

#define SEED_DELIMITER      ';'
#define SEED_POS_LIST_DELIMITER  ':'
#define SEED_POS_DELIMITER_1  ' '
#define SEED_POS_DELIMITER_2  ','
#define SEED_POS_DELIMITER    ", "

#ifdef NUCLEOTIDES
#define DEFAULT_SEED_FILE "seeds/solexa_lossy-3-12.in"
#else
#define DEFAULT_SEED_FILE "seeds/solid_lossy-3-12.in"
#endif

/* In case something happened to the file ... */
#ifdef NUCLEOTIDES
#define DEFAULT_SEEDS "####--#-#--##--####;#####--###-#-###;###-##-#######"
#else
#define DEFAULT_SEEDS "####----##--##-####;####-###--#----####;####-####-####"
#endif

typedef CODE_TYPE SeedSymbolType;

/**
 * @struct SeedType
 * describes a spaced seed
 */
typedef struct SeedType {
  /** seed weight (number of '1' or '#') */
  int weight;
  /** seed span */
  int length;
  /** seed symbols @see SeedSymbolType and its macros */
  SeedSymbolType* seed;
  /** list of 0 starting positions for the seed where a '#' occurs, and its size */
  /** @{ */
  int* positions;
  int positions_count;
  /** @} */
  /** binary mask (bit set to 1 when maching symbol, 0 otherwise) */
  unsigned long long int mask;
} SeedType;

/**
 * Create a seed from a pattern
 * @param pattern Accepted forms: "1        001..." or "#-##--#..."
 * @return The address of the created seed.
 */
SeedType* seed__create(const char* pattern);

/**
 * Creates an array of seeds from an input string of the form
 * <seed_pattern>(<seed_delimiter><seed_pattern>)*
 * @param src The input string
 * @param seeds Adress where the resulted array is put
 * @return The number of created seeds
 */
int seed__parse_list(const char* src, SeedType*** seeds);

/**
 * Read a seed list from a file
 * @param filename
 * @return the string of the seed in final ";" format
 */
char* seed__string_from_file(const char* filename);

/**
 * Apply a seed to the given sequence
 * @param seed
 * @param sequence
 * @param pos
 * @return The code of the indexed zone
 */
int seed__apply(const SeedType *seed, const CODE_TYPE* sequence, const int pos);

/**
 * Apply a seed to the given compressed sequence
 * @param seed
 * @param sequence
 * @param pos
 * @return The code of the indexed zone
 */
int seed__apply_to_compressed(const SeedType *seed, const CODE_TYPE* sequence, const int pos);

/**
 * Generate a friendly representation of the seed
 * @param seed
 * @return A string holding the representation
 */
char* seed__to_string(const SeedType *seed);

/**
 * Display a seed
 * @param seed
 */
void seed__display(const SeedType *seed);

/**
 * Destroy a seed
 * @param seed
 */
void seed__destroy(SeedType* seed);


#endif /* _SEED_H_ */
