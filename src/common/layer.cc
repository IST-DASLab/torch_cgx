#include "layer.h"

namespace cgx {
namespace common {

Layer::Layer(const at::Tensor &tensor) {
  name_ = "";
  data_ = tensor.data_ptr();
  element_size_ = tensor.element_size();
  numel_ = tensor.numel();
  scalar_type_ = tensor.scalar_type();
  device_index_ = tensor.get_device();
}

Layer::Layer(const at::Tensor &tensor, const std::string &layer_name,
             void *ptr, int numel) {
  name_ = layer_name;
  data_ = ptr;
  numel_ = numel;
  element_size_ = tensor.element_size();
  scalar_type_ = tensor.scalar_type();
  device_index_ = tensor.get_device();
}

} // namespace common
} // namespace cgx