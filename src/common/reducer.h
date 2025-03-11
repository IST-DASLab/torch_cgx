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

#include "common.h"
#include "compressor.h"
#include "gpu_context.h"
#include "utils.h"
#include <mpi.h>
#include <utility>

#include "mpi_communicator.h"
#if HAVE_CUDA
#include "shm_communicator.h"
#endif

namespace cgx::common {

class Reducer {
public:
  Reducer(std::shared_ptr<common::GPUContext> gpu_context,
          std::shared_ptr<Compressor> compressor,
          std::shared_ptr<Communicator> communicator= nullptr);

  virtual ~Reducer() = default;
  virtual int AllreduceDivision(int num_elements, int global_offset,
                                std::vector<Layer> &layers,
                                void *comm, gpuStream_t gpu_stream,
                                bool do_compression) = 0;
  virtual int AllReduceAlltoAll(int num_elements, int global_offset,
                    std::vector<Layer> &layers,
                    void *comm, gpuStream_t gpu_stream);
  virtual int Broadcast(int num_elements, int global_offset,
                std::vector<Layer> &layers,
                void *comm, gpuStream_t gpu_stream, bool do_compression);
protected:
  std::shared_ptr<Compressor> compressor_;
  std::shared_ptr<Communicator> communicator_;
  std::shared_ptr<common::GPUContext> gpu_context_;

  // We only need some framework agnostic Buffer Manager so we reuse
  // FussionBufferManager. Our usage of it is not related to tensor fusion
  // buffer.
  std::unique_ptr<PersistentBuffer> buffer_;
  unsigned char *gradients_send_ = nullptr;
  unsigned char *gradients_recv_ = nullptr;
  size_t tensor_fusion_size_;
};

class MPIReducer : public Reducer {
public:
  MPIReducer(std::shared_ptr<common::GPUContext> gpu_context,
             std::shared_ptr<Compressor> compressor,
             std::shared_ptr<Communicator> communicator)
      : Reducer(std::move(gpu_context), std::move(compressor), std::move(communicator)) {}
};

void printDebug(unsigned char *buf, int numel);
} // namespace cgx::common
