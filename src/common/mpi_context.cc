#include "mpi_context.h"

namespace cgx {
namespace common {

MPIContext::MPIContext() {
  MPI_Comm_dup(MPI_COMM_WORLD, &global_comm_);
  MPI_Comm_split_type(global_comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &local_comm_);
  int local_rank, world_rank;
  MPI_Comm_rank(global_comm_, &world_rank);
  MPI_Comm_rank(local_comm_, &local_rank);

  // Create cross node communicator.
  MPI_Comm_split(global_comm_, local_rank, world_rank, &cross_comm_);
}

int MPIContext::GetSize(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

int MPIContext::GetRank(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

} // namespace common
} // namespace cgx
