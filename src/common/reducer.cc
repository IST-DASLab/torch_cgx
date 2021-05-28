#include "reducer.h"
namespace qmpi {
namespace common {

Reducer::Reducer(GPUContext *gpu_context,
                 std::shared_ptr<Compressor> compressor)
    : compressor_(compressor), gpu_context_(gpu_context) {
  int fusion_size_mb =
      utils::GetIntEnvOrDefault(FUSION_BUFFER_SIZE_MB, FUSION_SIZE_DEFAULT_MB);
  tensor_fusion_size_ = fusion_size_mb * 1024 * 1024;
}

} // namespace common
} // namespace qmpi


