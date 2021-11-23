#pragma once

#include <torch/torch.h>

namespace cgx {
namespace common {

struct Layer {
  Layer(const at::Tensor& tensor);
  Layer(const at::Tensor& tensor, const std::string& layer_name,
        void* ptr, int numel);
  void* data_ptr() const {return data_;}
  int numel() const { return numel_; }
  size_t element_size() const {return element_size_;};
  at::ScalarType scalar_type() const { return scalar_type_;}
  const std::string& name() const { return name_; }
  const int64_t& device_index() const { return device_index_; }
private:
  std::string name_;
  void* data_;
  int numel_;
  size_t element_size_;
  at::ScalarType scalar_type_;
  int64_t device_index_;
};

} // namespace common
} // namespace cgx
