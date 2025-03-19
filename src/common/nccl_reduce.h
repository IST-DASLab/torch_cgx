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
#include <nccl.h>

namespace cgx::common {

class NCCL_Reduce : public Reducer {
public:
  NCCL_Reduce(std::shared_ptr<GPUContext> gpu_context,
              std::shared_ptr<Compressor> compressor, int world_size);
  int AllReduceAlltoAll(int num_elements, int global_offset,
                    std::vector<Layer> &layers, void *comm_p,
                    gpuStream_t gpu_stream) override;
  int AllreduceDivision(int num_elements, int global_offset,
                        std::vector<Layer> &tensors, void *comm,
                        gpuStream_t gpu_stream, bool do_compression) override;
  int Broadcast(int num_elements, int global_offset, std::vector<Layer> &layers,
                void *comm, gpuStream_t gpu_stream,
                bool do_compression) override;

private:
  void Init(void *comm);
  void ErrorCheck(const char* op_name, ncclResult_t nccl_result);
  void FuseLayerData(unsigned char **layers_data, std::vector<Layer> &layers,
                     int num_elements, int global_offset, gpuStream_t stream);
  void UnfuseLayerData(unsigned char *layers_data, std::vector<Layer> &layers,
                       int num_elements, gpuStream_t stream);

  int AllreduceUncompressed(int num_elements, int global_offset,
                            std::vector<Layer> &tensors, void *comm,
                            gpuStream_t gpu_stream);

  int AllreduceCompressed(int num_elements, int global_offset,
                          std::vector<Layer> &tensors, void *comm,
                          gpuStream_t gpu_stream);

  ncclComm_t nccl_comm_;
};

} // namespace cgx::common
