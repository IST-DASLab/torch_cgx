#pragma once
#include <cstdlib>
#include <torch/torch.h>

namespace qmpi {
namespace common {
namespace utils {
const size_t ALIGNMENT_UNIT = 2 * sizeof(float);

int GetIntEnvOrDefault(const char *env_variable, int default_value);

void SetBoolFromEnv(const char* env, bool& val, bool value_if_set);

size_t get_sizeof(at::ScalarType dtype);

size_t round_to(size_t x, size_t m);

size_t aligned_size(size_t size);


} // namespace utils
} // namespace common
} // namespace qmpi