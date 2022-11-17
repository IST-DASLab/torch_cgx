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

#include <torch/torch.h>
#include <utility>

namespace cgx {
namespace common {
using LayerId = std::pair<unsigned, unsigned>;

struct Layer {
  Layer(const at::Tensor& tensor);
  Layer(const at::Tensor& tensor, const LayerId& layer_id,
        void* ptr, int numel);
  const LayerId& layer_id() const {return layer_id_;}
  void* data_ptr() const {return data_;}
  int numel() const { return numel_; }
  size_t element_size() const {return element_size_;};
  at::ScalarType scalar_type() const { return scalar_type_;}
  const int64_t& device_index() const { return device_index_; }
private:
  LayerId layer_id_;
  void* data_;
  int numel_;
  size_t element_size_;
  at::ScalarType scalar_type_;
  int64_t device_index_;
};

} // namespace common
} // namespace cgx
