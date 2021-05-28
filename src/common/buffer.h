#pragma once
#include <torch/torch.h>

namespace qmpi {
namespace common {

struct PersistentBuffer {
  PersistentBuffer(size_t size);
  void* RawPointer() const;
private:
  at::Tensor tensor_;
};


} // namespace common
} // namespace qmpi
