## Tested on 
# - gcc 4.8 (sse2,avx2) [linux]
# - gcc-mp-4.8 + llvm-gcc-4.2 (sse2) + icc 14 (sse2,avx2) [mac]
CC      = gcc

## Without the "-msse2" or "-mavx2", the program is more than ten time slower
CFLAGS  = -Wall -pipe -Ofast -msse2
## Replace with "-mavx2" for processors that are at least >= mid-2013 and support the "avx2" ?
##  - For Linux : cat /proc/cpuinfo | grep avx2
##  - For Mac   : sysctl -a | grep avx2
#CFLAGS  = -Wall -pipe -mavx2 -O3

## For large files support :
CFLAGS += -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE 

## For OpenMP support :
CFLAGS += -fopenmp

LDFLAGS = -lm

## Source :
ALL_C   = main.c alignment.c alignment_simd.c codes.c hit_map.c \
   index.c load_data.c genome_map.c reads_data.c read_quality.c \
   reference_data.c run.c seed.c stats.c util.c

##  SToRM has been build to work in color space by default (SOLiD reads), 
##  but is now also able to work in full nucleotide space (Illumina reads) 
##  (Note that it only supports fixed length reads)

all: storm-color storm-nucleotide

storm-color :
	${CC} ${ALL_C} -o storm-color ${CFLAGS} ${LDFLAGS}

storm-nucleotide :
	${CC} -DNUCLEOTIDES ${ALL_C} -o storm-nucleotide ${CFLAGS} ${LDFLAGS}

clean:
	rm -f *~ storm-color storm-nucleotide
