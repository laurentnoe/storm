#ifndef _LOAD_DATA_H_
#define _LOAD_DATA_H_

#include "codes.h"
#include "read_quality.h"

#define RETURN_SUCCESS 0
#define RETURN_INPUT_ERR (-1)
#define RETURN_MEM_ERR (-2)

#define LINE_TYPE_COMMENT     0
#define LINE_TYPE_SEQ_HEADER  1
#define LINE_TYPE_SEQ         2
#define LINE_TYPE_QUAL_HEADER 3
#define LINE_TYPE_QUAL        4
#define LINE_TYPE_UNKNOWN   (-1)

/* Length of the line read from the input files */
#define MAX_LINE_LEN 65536

#define SEQ_HEADER_MARKER '>'
#define SEQ_HEADER_MARKER_FASTQ '@'
#define SEQ_QUALITY_MARKER_FASTQ '+'
#define COMMENT_MARKER    '#'

int parse_integer_sequence_to(const char* src, short* dest, int len);

short* parse_integer_sequence(const char* src, int len) ;

int get_input_line_type(char* line);

int get_input_line_type_fastq(char* line, int lineNo, int offset);

void delete_trailing_nl(char* line) ;

#endif /* _LOAD_DATA_H_ */
