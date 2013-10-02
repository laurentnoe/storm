#include "load_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Populates a short integer array from a string that represents an integer enumeration, delimited by " ".
 * Useful for parsing read quality files and command line read quality parameters.
 * @param src The source string to be parsed
 * @param dest The destination vector, already allocated.
 * @param len The expected number of integers.
 * @return 0 if all the numbers were successfully read, 1 if the source finished prematurely.
 */
int parse_integer_sequence_to(const char* src, short* dest, int len) {
  char *str1, *token = NULL, *delim = " ", *src_copy;
  int j = 0;
  SAFE_FAILURE__ALLOC(src_copy, strlen(src) + 1, char);
  memset(src_copy,'\0',strlen(src) + 1);
  if (src_copy) {
    strcpy(src_copy, src);

    for (j = 0, str1 = src_copy; j < len ; j++, str1 = NULL) {
      token = strtok(str1, delim);
      if (token == NULL) {
        break;
      }
      dest[j] = atoi(token);
    }
    free(src_copy);
  }
  return (j < len) ? RETURN_INPUT_ERR : RETURN_SUCCESS;
}
/**
 * Creates and populates a short integer array from a string that represents an integer enumeration, delimited by " ".
 * Useful for parsing read quality files and command line read quality parameters.
 * @param src The source string to be parsed
 * @param len The expected number of integers.
 * @return 0 if all the numbers were successfuly read, 1 if the source finished prematurely.
 */
short* parse_integer_sequence(const char* src, int len) {
  short* result;
  SAFE_FAILURE__ALLOC(result, len, short);
  memset(result,'\0',len);
  parse_integer_sequence_to(src, result, len);
  return result;
}

/**
 * Determines what kind of input was read. The input files are expected to start with comments marked by '#'
 * as the first character in the line (comments that can be missing), followed by one or more sequences, each
 * preceded by some header information (marked by '>' as the first character).
 * @param line The string that was read
 * @return The kind of information it corresponds to. Possible values:
 * LINE_TYPE_COMMENT, LINE_TYPE_SEQ_HEADER, LINE_TYPE_SEQ, LINE_TYPE_UNKNOWN.
 */
int get_input_line_type(char* line) {
  static int crt_line_type = LINE_TYPE_UNKNOWN;
  int line_type;
  if ((line[0] == '\n') || (line[0] == '\r') || (line[0] == '\0')) {
    return (crt_line_type = LINE_TYPE_UNKNOWN);
  } else if (crt_line_type == LINE_TYPE_COMMENT || (line[0] == COMMENT_MARKER)) {
    line_type = LINE_TYPE_COMMENT;
  } else if (crt_line_type == LINE_TYPE_SEQ_HEADER || (line[0] == SEQ_HEADER_MARKER)) {
    line_type = LINE_TYPE_SEQ_HEADER;
  } else {
    line_type = LINE_TYPE_SEQ;
  }
  // if the input does not end with newline or the input is a sequence, the next input will be of the same type as the current
  crt_line_type = ((line_type == LINE_TYPE_SEQ) || (line[strlen(line) - 1] != '\n')) ? line_type : LINE_TYPE_UNKNOWN ;
  //printf("\n[%s] --> line type=%d, crt_line_type=%d\n", line, line_type, crt_line_type);
  return line_type;
}

/**
 * Determines what kind of input was read (from a fastq file). The input files are expected to start with comments marked by '#'
 * as the first character in the line (comments that can be missing), followed by one or more sequences, each
 * preceded by some header information (marked by '@' as the first character) and their quality (marked by '+').
 * @param line The string that was read
 * @param lineNo The line numero
 * @param offset number of commented lines from the beginning.
 * @return The kind of information it corresponds to. Possible values:
 * LINE_TYPE_COMMENT, LINE_TYPE_SEQ_HEADER, LINE_TYPE_SEQ, LINE_TYPE_QUAL_HEADER, LINE_TYPE_QUAL, LINE_TYPE_UNKNOWN.
 */
int get_input_line_type_fastq(char* line, int lineNo, int offset) {
  static int crt_line_type = LINE_TYPE_UNKNOWN;
  int line_type;
  if ((line[0] == '\n') || (line[0] == '\r') || (line[0] == '\0')) {
    return LINE_TYPE_UNKNOWN;
  } else if (crt_line_type == LINE_TYPE_COMMENT || (line[0] == COMMENT_MARKER)) {
    line_type = LINE_TYPE_COMMENT;
  } else if ((lineNo - offset) % 4 == 0 && line[0] == SEQ_HEADER_MARKER_FASTQ) {
    line_type = LINE_TYPE_SEQ_HEADER;
  } else if ((lineNo - offset) % 4 == 2 && line[0] == SEQ_QUALITY_MARKER_FASTQ) {
    line_type = LINE_TYPE_QUAL_HEADER;
  } else if ((lineNo - offset) % 4 == 1 && crt_line_type == LINE_TYPE_SEQ_HEADER){
    line_type = LINE_TYPE_SEQ;
  } else if ((lineNo - offset) % 4 == 3 && crt_line_type == LINE_TYPE_QUAL_HEADER){
    line_type = LINE_TYPE_QUAL;
  } else {
    line_type = LINE_TYPE_UNKNOWN;
  }
  // if the input does not end with newline or the input is a sequence, the next input will be of the same type as the current
  crt_line_type = ((line_type == LINE_TYPE_SEQ) || (line_type == LINE_TYPE_SEQ_HEADER) ||
                  (line_type == LINE_TYPE_QUAL)  || (line_type == LINE_TYPE_QUAL_HEADER))
                  ? line_type : LINE_TYPE_UNKNOWN ;
  //printf("\n[%s] --> line type=%d, crt_line_type=%d\n", line, line_type, crt_line_type);
  return line_type;
  }

/**
 * If the last character from the string is '\n', this method replaces it with '\0'.
 */
inline void delete_trailing_nl(char* line) {
  int len;
  while ((len = strlen(line)) > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[len - 1] = '\0';
}
