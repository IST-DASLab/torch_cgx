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

#include "buffer.h"

namespace cgx {
namespace common {
PersistentBuffer::PersistentBuffer(size_t size) {
//  tensor_ = at::empty(size, at::device(at::kCUDA).dtype(at::kByte));
//  tensor_.zero_();
  tensor_ = at::zeros(size, at::device(at::kCUDA).dtype(at::kByte));
}

void * PersistentBuffer::RawPointer() const {
  return tensor_.data_ptr();
}

} // namespace common
} // namespace cgx

