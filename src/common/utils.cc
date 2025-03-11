/*
 * pytorch-cgx
 *
 * Copyright (C) 2022 Institute of Science and Technology Austria (ISTA).
 * All Rights Reserved.
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "utils.h"
#include <string>

namespace cgx::common::utils {

int GetIntEnvOrDefault(const char *env_variable, int default_value) {
  auto env_value = std::getenv(env_variable);
  return env_value != nullptr ? std::strtol(env_value, nullptr, 10)
                              : default_value;
}

float GetFloatEnvOrDefault(const char *env_variable, float default_value) {
  auto env_value = std::getenv(env_variable);
  return env_value != nullptr ? std::stof(std::string(env_value))
                              : default_value;
}

void SetBoolFromEnv(const char *env, bool &val, bool value_if_set) {
  auto env_value = std::getenv(env);
  if (env_value != nullptr && std::strtol(env_value, nullptr, 10) > 0) {
    val = value_if_set;
  }
}

CommunicatorType GetCommTypeFromEnv(const char* env, CommunicatorType default_value) {
  auto env_value = std::getenv(env);
  if (env_value == nullptr)
    return default_value;
  auto env_value_str = std::string(env_value);
  if (env_value_str == "MPI")
    return CommunicatorType::MPI;
  else if (env_value_str == "SHM")
    return CommunicatorType::SHM;
  else if (env_value_str == "NCCL")
    return CommunicatorType::NCCL;
  else
    throw std::runtime_error("Unknown type of communicator");
}

ReductionType GetRedTypeFromEnv(const char* env, ReductionType default_value) {
  auto env_value = std::getenv(env);
  if (env_value == nullptr)
    return default_value;
  auto env_value_str = std::string(env_value);
  if (env_value_str == "SRA")
    return ReductionType::SRA;
  else if (env_value_str == "Ring")
    return ReductionType::Ring;
  else
    throw std::runtime_error("Unknown type of reduction");
}

#ifndef TEST
size_t get_sizeof(at::ScalarType dtype) {
  if (dtype == at::kHalf) {
    return sizeof(float) / 2;

  } else if (dtype == at::kFloat) {
    return sizeof(float);
  } else {
    throw std::runtime_error("Unknown type at get_sizeof");
  }
}
#endif

size_t round_to(size_t x, size_t m) {
  return x + ((m - x % m) % m);
}

size_t aligned_size(size_t size) {
  return round_to(size, ALIGNMENT_UNIT);
}

} // namespace cgx::common::utils
