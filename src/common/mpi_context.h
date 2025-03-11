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
#include <mpi.h>

namespace cgx::common {

struct MPIContext {
  MPIContext();
  MPI_Comm GetGlobalComm() const {return global_comm_;}
  MPI_Comm GetLocalComm() const {return local_comm_;}
  MPI_Comm GetCrossComm() {return cross_comm_;}
  int GetRank(MPI_Comm comm) const;
  int GetSize(MPI_Comm comm) const;
  void Barrier(MPI_Comm comm) const;
private:
  MPI_Comm global_comm_;
  MPI_Comm local_comm_;
  MPI_Comm cross_comm_;
};

} // namespace cgx::common
