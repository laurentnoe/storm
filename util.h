#ifndef _UTIL_H_
#define _UTIL_H_

#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>

#define PROGRAM_NAME    "SToRM"
#define PROGRAM_VERSION "0.0099"

/* flushed write to trace */
#define __W {fprintf(stderr, "\033[35;1m%s %d\033[0m\n", __FILE__, __LINE__); fflush(stderr);}

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ABS(a)    ((a) >= 0  ? (a) :-(a))

#define VERBOSITY_NONE     0
#define VERBOSITY_MODERATE 1
#define VERBOSITY_HIGH     2
#define VERBOSITY_ANNOYING 3
#define VERBOSITY_MAX      VERBOSITY_ANNOYING
#define VERBOSITY_DEFAULT  VERBOSITY_MODERATE

#define DEFAULT_MAX_THREADS 8

/*
 * Number of positions kept per read in the hit_map structure
 */
#define MAP_DETAIL_SIZE 4

extern int VERBOSITY;

#ifdef _OPENMP
extern int MAX_THREADS;
#endif

/* alignment sense selection */
extern int ALIGNMENT_SENSE;
#define ALIGNMENT_SENSE_FORWARD 0x1
#define ALIGNMENT_SENSE_REVERSE 0x2
#define DEFAULT_ALIGNMENT_SENSE (ALIGNMENT_SENSE_FORWARD | ALIGNMENT_SENSE_REVERSE)

/* verbosity macro and colored messages */
#define VERB_FILTER(verbosity_level, f) if ((verbosity_level) <= VERBOSITY) { f; }

#ifdef _OPENMP

#define OMP_INTERNAL_CRITICAL_PROGRESS _Pragma("omp critical(progress)")
#define ERROR__(...)    OMP_INTERNAL_CRITICAL_PROGRESS {fprintf(stderr,"\033[31;1m"); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\033[0m\n"); fflush(NULL); }
#define WARNING__(...)  OMP_INTERNAL_CRITICAL_PROGRESS {fprintf(stderr,"\033[33;1m"); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\033[0m\n"); fflush(NULL); }
#define INFO__(...)     OMP_INTERNAL_CRITICAL_PROGRESS {fprintf(stderr,"\033[32;1m"); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\033[0m\n"); fflush(NULL); }
#define DEBUG__(...)    OMP_INTERNAL_CRITICAL_PROGRESS {fprintf(stderr,"\033[35;1m"); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\033[0m\n"); fflush(NULL); }
#define MESSAGE__(...)  OMP_INTERNAL_CRITICAL_PROGRESS {                              fprintf(stderr,__VA_ARGS__);                              fflush(NULL); }

#else

#define ERROR__(...)   {fprintf(stderr,"\033[31;1m"); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\033[0m\n"); fflush(NULL); }
#define WARNING__(...) {fprintf(stderr,"\033[33;1m"); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\033[0m\n"); fflush(NULL); }
#define INFO__(...)    {fprintf(stderr,"\033[32;1m"); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\033[0m\n"); fflush(NULL); }
#define DEBUG__(...)   {fprintf(stderr,"\033[35;1m"); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\033[0m\n"); fflush(NULL); }
#define MESSAGE__(...) {                              fprintf(stderr,__VA_ARGS__);                              fflush(NULL); }

#endif




/* allocation / aligned allocation (SSE) / reallocation */
#define SAFE_FAILURE__ALLOC(_ptr,_size,_unit_type) {                               \
  (_ptr) = (_unit_type*)malloc((size_t)(_size)*(size_t)sizeof(_unit_type));        \
  if ((_ptr) == NULL) {                                                            \
    ERROR__( "\nNot enough available memory.\n%s:%d  %zu X %zu = %zu bytes needed" \
             "\n\nExiting.\n", __FILE__, __LINE__,                                 \
             (size_t)(_size),                                                      \
             (size_t)sizeof(_unit_type),                                           \
             (size_t)(_size) * sizeof(_unit_type));                                \
             exit(1);                                                              \
  }                                                                                \
}

extern int N_BYTES;

#define SAFE_FAILURE__ALLOC_ALIGNED(_ptr,_size,_unit_type) {                        \
  posix_memalign(&_prt,N_BYTES,(size_t)(_size)*(size_t)sizeof(_unit_type));         \
  if ((_ptr) == NULL) {                                                             \
    ERROR__( "\nNot enough available memory.\n%s:%d  %zu X %zu = %zu bytes needed"  \
             "\n\nExiting.\n", __FILE__, __LINE__,                                  \
             (size_t)(_size),                                                       \
             (size_t)sizeof(_unit_type),                                            \
             (size_t)(_size) * sizeof(_unit_type));                                 \
             exit(1);                                                               \
  }                                                                                 \
}


#define SAFE_FAILURE__REALLOC(_ptr,_size,_unit_type) {                             \
  (_ptr) = (_unit_type*)realloc(_ptr,(size_t)(_size)*(size_t)sizeof(_unit_type));  \
  if ((_ptr) == NULL) {                                                            \
    ERROR__( "\nNot enough available memory.\n%s:%d  %zu X %zu = %zu bytes needed" \
             "\n\nExiting.\n", __FILE__, __LINE__,                                 \
             (size_t)(_size),                                                      \
             (size_t)sizeof(_unit_type),                                           \
             (size_t)(_size) * sizeof(_unit_type));                                \
             exit(1);                                                              \
  }                                                                                \
}




/**
 * Displays a progress bar that shows the percentage of processed data
 * @param p The variable part for the progress bar
 * @param total The maximal value that p has to reach
 * @param selected A selected part of p that is printed (as a percentage of total) if not set to 0
 */
void display_progress(long int p, long int total, long int selected);

/**
 * Sort the first array (values) without changing it and put the indices in the right order
 * in the second array (result_indices).
 * @param values The values that need to be sorted
 * @param result_indices The array holding the indices of the sorted values
 * @param size The size of the two arrays
 */
void sort__indices(const int* values, int* result_indices, const int size);

#endif /* _UTIL_H_ */


