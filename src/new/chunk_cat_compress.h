#include <torch/torch.h>

#ifndef TORCH_CGX_CHUNK_CAT_COMPRESS_H
#define TORCH_CGX_CHUNK_CAT_COMPRESS_H

// torch.ops.fsdp.chunk_cat(
//        unsharded_grads, dim=0, num_chunks=world_size, out=reduce_scatter_input
//    )

void chunk_cat_compress(const std::vector<at::Tensor>& inputs, at::Tensor& output, at::Tensor& meta) {

}

#endif //TORCH_CGX_CHUNK_CAT_COMPRESS_H
