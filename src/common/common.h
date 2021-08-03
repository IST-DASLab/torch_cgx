#pragma once

#define FUSION_BUFFER_SIZE_MB "FUSION_BUFFER_SIZE_MB"
#define COMPRESSION_MINIMAL_SIZE "COMPRESSION_MINIMAL_SIZE"
#define COMPRESSION_QUANTIZATION_BITS "COMPRESSION_QUANTIZATION_BITS"
#define COMPRESSION_BUCKET_SIZE "COMPRESSION_BUCKET_SIZE"
#define COMPRESSION_SKIP_INCOMPLETE_BUCKETS "COMPRESSION_SKIP_INCOMPLETE_BUCKETS"
#define COMPRESSION_FAKE_RATIO "COMPRESSION_FAKE_RATIO"

const unsigned int FUSION_SIZE_DEFAULT_MB = 64;
const unsigned int MIN_FUSION_SIZE = 2048;

#define MPI_CHECK(condition)                                                   \
  do {                                                                         \
    int op = condition;                                                        \
    if (op != MPI_SUCCESS) {                                                   \
      int len;                                                                 \
      char estring[MPI_MAX_ERROR_STRING];                                      \
      MPI_Error_string(op, estring, &len);                                     \
      printf("%s on line %i. MPI Error: %s\n", #condition, __LINE__, estring); \
      throw std::runtime_error(std::string(#condition) + " on line " +         \
                               std::to_string(__LINE__) + " failed: ");        \
    }                                                                          \
  } while (0)
