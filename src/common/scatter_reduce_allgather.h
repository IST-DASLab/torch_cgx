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
#include "reducer.h"
#include "communicator.h"

namespace cgx {
namespace common {

class MPI_Allreduce_ScatterReduceAllgather : public MPIReducer {
public:
  MPI_Allreduce_ScatterReduceAllgather(std::shared_ptr<GPUContext> gpu_context,
                                       std::shared_ptr<Compressor> compressor,
                                       std::shared_ptr<Communicator> communicator,
                                       int world_size);

  int AllreduceDivision(int num_elements, int global_offset,
                        std::vector<Layer> &tensors,
                        void *comm, gpuStream_t gpu_stream, bool do_compression) override;
private:
  int AllreduceCompressed(int num_elements, int global_offset,
                        std::vector<Layer> &layers,
                        void *comm, gpuStream_t gpu_stream);
  int AllreduceUncompressed(int num_elements, int global_offset,
                        std::vector<Layer> &layers,
                        void *comm, gpuStream_t gpu_stream);
  int AllreduceCompressedRemoteBuf(int num_elements,
                               int global_offset,
                               std::vector<
                                   Layer> &layers,
                               void *comm_p, gpuStream_t gpu_stream);
  int AllReduceAlltoAllCompressed(int num_elements, int global_offset,
                        std::vector<Layer> &layers,
                        void *comm, gpuStream_t gpu_stream);
private:
  bool remote_buf_compression_enabled_;
  bool all_to_all_reduction_;
  unsigned counter_ = 0;
};

} // namespace common
} // namespace cgx
