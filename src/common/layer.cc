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

#include "layer.h"

namespace cgx {
namespace common {

Layer::Layer(const at::Tensor &tensor) {
  data_ = tensor.data_ptr();
  element_size_ = tensor.element_size();
  numel_ = tensor.numel();
  scalar_type_ = tensor.scalar_type();
  device_index_ = tensor.get_device();
  layer_id_ = std::make_pair(0, 0);
}

Layer::Layer(const at::Tensor &tensor, const LayerId &layer_id,
             void *ptr, int numel) {
  layer_id_ = layer_id;
  data_ = ptr;
  numel_ = numel;
  element_size_ = tensor.element_size();
  scalar_type_ = tensor.scalar_type();
  device_index_ = tensor.get_device();
}

} // namespace common
} // namespace cgx