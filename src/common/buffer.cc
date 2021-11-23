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

