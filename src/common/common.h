#pragma once

#define CGX_FUSION_BUFFER_SIZE_MB "CGX_FUSION_BUFFER_SIZE_MB"
#define CGX_COMPRESSION_MINIMAL_SIZE "CGX_COMPRESSION_MINIMAL_SIZE"
#define CGX_COMPRESSION_QUANTIZATION_BITS "CGX_COMPRESSION_QUANTIZATION_BITS"
#define CGX_COMPRESSION_BUCKET_SIZE "CGX_COMPRESSION_BUCKET_SIZE"
#define CGX_COMPRESSION_SKIP_INCOMPLETE_BUCKETS "CGX_COMPRESSION_SKIP_INCOMPLETE_BUCKETS"
#define CGX_COMPRESSION_FAKE_RATIO "CGX_COMPRESSION_FAKE_RATIO"
#define CGX_DUMMY_COMPRESSION "CGX_DUMMY_COMPRESSION"
#define CGX_INNER_COMMUNICATOR_TYPE "CGX_INNER_COMMUNICATOR_TYPE"
#define CGX_INNER_REDUCTION_TYPE "CGX_INNER_REDUCTION_TYPE"
#define CGX_CROSS_REDUCTION_TYPE "CGX_CROSS_REDUCTION_TYPE"
#define CGX_REMOTE_BUF_COMPRESSION "REMOTE_BUF_COMPRESSION"
#define CGX_DEBUG_ALL_TO_ALL_REDUCTION "DEBUG_ALL_TO_ALL_REDUCTION"

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
