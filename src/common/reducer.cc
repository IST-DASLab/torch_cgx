#include "reducer.h"
namespace cgx {
namespace common {

Reducer::Reducer(GPUContext *gpu_context,
                 std::shared_ptr<Compressor> compressor,
                 std::shared_ptr<Communicator> communicator)
    : compressor_(compressor),
      gpu_context_(gpu_context),
      communicator_(communicator) {
  unsigned int fusion_size_mb =
      utils::GetIntEnvOrDefault(CGX_FUSION_BUFFER_SIZE_MB, FUSION_SIZE_DEFAULT_MB);
  tensor_fusion_size_ = std::max(fusion_size_mb * 1024 * 1024, MIN_FUSION_SIZE);
}

} // namespace common
} // namespace cgx


