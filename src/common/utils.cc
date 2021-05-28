#include "utils.h"

namespace qmpi {
namespace common {
namespace utils {

int GetIntEnvOrDefault(const char *env_variable, int default_value) {
  auto env_value = std::getenv(env_variable);
  return env_value != nullptr ? std::strtol(env_value, nullptr, 10)
                              : default_value;
}

void SetBoolFromEnv(const char *env, bool &val, bool value_if_set) {
  auto env_value = std::getenv(env);
  if (env_value != nullptr && std::strtol(env_value, nullptr, 10) > 0) {
    val = value_if_set;
  }
}

size_t get_sizeof(at::ScalarType dtype) {
  if (dtype == at::kHalf) {
    return sizeof(float) / 2;

  } else {
    return sizeof(float);
  }
}

size_t round_to(size_t x, size_t m) {
  return x + ((m - x % m) % m);
}

size_t aligned_size(size_t size) {
  return round_to(size, ALIGNMENT_UNIT);
}

} // namespace utils
} // namespace common
} // namespace qmpi