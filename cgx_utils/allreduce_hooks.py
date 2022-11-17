#  pytorch-cgx
#
#  Copyright (C) 2022 IST Austria
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from typing import Dict
import torch
import torch.distributed as dist
import torch_cgx
import os

COMPRESSION_QUANTIZATION_BITS = "CGX_COMPRESSION_QUANTIZATION_BITS"
COMPRESSION_BUCKET_SIZE = "CGX_COMPRESSION_BUCKET_SIZE"
COMPRESSION_MINIMAL_SIZE = "CGX_COMPRESSION_MINIMAL_SIZE"
VALUE_NO_COMPRESS=32


class CGXState(object):
    def __init__(self, process_group: dist.ProcessGroup, layer_min_size: int = 1024,
                 compression_params: Dict[str, int] = None):
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        min_size_to_compress = int(os.getenv(COMPRESSION_MINIMAL_SIZE, "16"))
        self.layer_min_size = max(layer_min_size, min_size_to_compress)
        self.quantization_bits = int(os.getenv(COMPRESSION_QUANTIZATION_BITS, str(VALUE_NO_COMPRESS)))
        self.quantization_bucket_size = int(os.getenv(COMPRESSION_BUCKET_SIZE, "1024"))
        self.step = 0
        if compression_params is not None:
            self.quantization_bits = compression_params.get("bits", self.quantization_bits)
            self.quantization_bucket_size = compression_params.get("bucket_size", self.quantization_bucket_size)

    def should_compress_(self, tensor: torch.Tensor):
        if tensor.dim() <= 1 or tensor.numel() < self.layer_min_size:
            return False
        return True


def _allreduce_fut(
        process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    "Averages the input gradient tensor by allreduce and returns a future."
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())
    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
    )


def cgx_hook(
        state: CGXState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    if state.step == 2:
        for layer_idx, tensor in enumerate(bucket.gradients()):
            bits = state.quantization_bits if state.should_compress_(tensor) else VALUE_NO_COMPRESS
            torch_cgx.register_layer(bucket.index(), layer_idx, tensor.numel(),
                                      bits, state.quantization_bucket_size)
    if bucket.is_last():
        state.step += 1
        state.layer_idx = 0
    return _allreduce_fut(state.process_group, bucket.buffer())
