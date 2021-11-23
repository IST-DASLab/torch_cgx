#pragma once
#include <cstdlib>
#include <torch/torch.h>

namespace cgx {
namespace common {
namespace utils {

enum CommunicatorType {
  MPI,
  SHM,
  P2P
};

enum ReductionType {
  SRA,
  Ring
};

const size_t ALIGNMENT_UNIT = 2 * sizeof(float);

int GetIntEnvOrDefault(const char *env_variable, int default_value);
float GetFloatEnvOrDefault(const char *env_variable, float default_value);

void SetBoolFromEnv(const char* env, bool& val, bool value_if_set);

size_t get_sizeof(at::ScalarType dtype);

size_t round_to(size_t x, size_t m);

size_t aligned_size(size_t size);

CommunicatorType GetCommTypeFromEnv(const char* env, CommunicatorType default_value);
ReductionType GetRedTypeFromEnv(const char* env, ReductionType default_value);

} // namespace utils
} // namespace common
} // namespace cgx