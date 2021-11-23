#pragma once
#include <mpi.h>

namespace cgx {
namespace common {

struct MPIContext {
  MPIContext();
  MPI_Comm GetGlobalComm() {return global_comm_;}
  MPI_Comm GetLocalComm() {return local_comm_;}
  MPI_Comm GetCrossComm() {return cross_comm_;}
  int GetRank(MPI_Comm comm);
  int GetSize(MPI_Comm comm);
private:
  MPI_Comm global_comm_;
  MPI_Comm local_comm_;
  MPI_Comm cross_comm_;
};

} // namespace common
} // namespace cgx


