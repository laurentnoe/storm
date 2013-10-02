#ifndef _STATS_H_
#define _STATS_H_

#include "genome_map.h"

/** Displays the traceback strings of the mapped reads (for statistics)
 */
void  display_tracebacks(GenomeMapType* genome_map) ;

/** Lists the unmapped reads
 */
void list_unmapped_reads_translated(GenomeMapType* genome_map);
void list_unmapped_reads(GenomeMapType* genome_map);


#endif /* _STATS_H_ */
