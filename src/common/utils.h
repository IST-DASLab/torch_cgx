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

#pragma once
#include <cstdlib>
#include <stdexcept>

#ifndef TEST
#include <torch/torch.h>
#endif

namespace cgx {
namespace common {
namespace utils {

enum CommunicatorType {
  MPI,
  SHM,
  NCCL
};

enum ReductionType {
  SRA,
  Ring
};

const size_t ALIGNMENT_UNIT = 2 * sizeof(float);

int GetIntEnvOrDefault(const char *env_variable, int default_value);
float GetFloatEnvOrDefault(const char *env_variable, float default_value);

void SetBoolFromEnv(const char* env, bool& val, bool value_if_set);

#ifndef TEST
size_t get_sizeof(at::ScalarType dtype);
#endif

size_t round_to(size_t x, size_t m);

size_t aligned_size(size_t size);

CommunicatorType GetCommTypeFromEnv(const char* env, CommunicatorType default_value);
ReductionType GetRedTypeFromEnv(const char* env, ReductionType default_value);

} // namespace utils
} // namespace common
} // namespace cgx