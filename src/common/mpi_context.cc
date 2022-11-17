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

#include "mpi_context.h"
#include "common.h"

namespace cgx {
namespace common {

MPIContext::MPIContext() {
  MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &global_comm_));
  MPI_CHECK(MPI_Comm_split_type(global_comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &local_comm_));
  int local_rank, world_rank;
  MPI_CHECK(MPI_Comm_rank(global_comm_, &world_rank));
  MPI_CHECK(MPI_Comm_rank(local_comm_, &local_rank));

  // Create cross node communicator.
  MPI_CHECK(MPI_Comm_split(global_comm_, local_rank, world_rank, &cross_comm_));
}

int MPIContext::GetSize(MPI_Comm comm) const {
  int size;
  MPI_CHECK(MPI_Comm_size(comm, &size));
  return size;
}

int MPIContext::GetRank(MPI_Comm comm) const {
  int rank;
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  return rank;
}

void MPIContext::Barrier(MPI_Comm comm) const {
  MPI_CHECK(MPI_Barrier(comm));
}

} // namespace common
} // namespace cgx
