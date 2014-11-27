#ifndef UTIL_H_

#include "util.h"
#include <stdio.h>

int VERBOSITY = VERBOSITY_DEFAULT;

#ifdef _OPENMP
int MAX_THREADS  = DEFAULT_MAX_THREADS;
#endif

int ALIGNMENT_SENSE = DEFAULT_ALIGNMENT_SENSE;

/**
 * Displays a progress bar that shows the percentage of processed data
 * @param p The variable part for the progress bar
 * @param total The maximal value that p has to reach
 * @param selected A selected part of p that is printed (as a percentage of total) if not set to 0
 */
inline void display_progress(long int p, long int total, long int selected) {
  static long long int perquantile = 0; /* to fluidify both progress bars : 300 = ppcm(100,60)*/
  static const char COMPLETED_SYMBOL = '=', UNCOMPLETED_SYMBOL = ' ', LAST_COMPLETED_SYMBOL = '>';
  if (p == 0 || ((long long int)(p + 1) * 300 / (total+1)) > perquantile) {
    long int      percent = (long long int)(p + 1) * 100 / (total+1);
    long int      perbar  = (long long int)(p + 1) * 60  / (total+1);
              perquantile = (long long int)(p + 1) * 300 / (total+1);
#ifdef _OPENMP
OMP_INTERNAL_CRITICAL_PROGRESS
#endif
    {
      fprintf(stderr,"\r\033[36;1m[");
      long int p2;
      for (p2 = 0; p2 < perbar; ++p2) {
        fprintf(stderr,"%c", COMPLETED_SYMBOL);
      }
      if (perbar && perbar != 60) {
        fprintf(stderr,"\b%c", LAST_COMPLETED_SYMBOL);
      }
      for (p2=perbar ; p2 < 60; ++p2) {
        fprintf(stderr,"%c", UNCOMPLETED_SYMBOL);
      }
      fprintf(stderr,"]");
      fprintf(stderr," %3d%%",  (int)percent);
      if (selected > 0) {
        fprintf(stderr,"  >> %12ld - %5.2lf%%", selected, (selected * 100.0 / (total+1)));
      }
      fprintf(stderr,"\033[0m");
      if (perbar == 60) {
        fprintf(stderr,"\n");
      }
      fflush(NULL);
    }
  }
}

/**
 * Sort the first array (values) without changing it and put the indices in the right order
 * in the second array (result_indices).
 * @param values The values that need to be sorted
 * @param result_indices The array holding the indices of the sorted values
 * @param size The size of the two arrays
 */
inline void sort__indices(const int* values, int* result_indices, const int size) {
  int i;
  for (i = 0; i < size; ++i) {
    result_indices[i] = i;
  }
  for (i = 1; i < size; ++i) {
    int val_idx = result_indices[i];
    int j = i - 1;
    do {
      if (values[result_indices[j]] < values[val_idx]) {
      result_indices[j + 1] = result_indices[j];
      --j;
      } else {
      break;
      }
    } while (j >= 0);
    result_indices[j + 1] = val_idx;
  }
}

#endif /* UTIL_H_ */
