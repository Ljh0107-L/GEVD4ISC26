#include "gevd/numerical_info.hpp"

#include "runtime/error.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <stdexcept>
#include <string>

namespace gevd4isc26::detail {

int readNumericalInfo(const DistributedGpuContext& context,
                      const int* device_info) {
  int host_info = 0;
  GEVD_CUDA(cudaMemcpyAsync(&host_info, device_info, sizeof(int),
                            cudaMemcpyDeviceToHost, context.stream()));
  context.synchronize();
  return host_info;
}

void checkNumericalInfo(const DistributedGpuContext& context,
                        int local_info,
                        const char* stage) {
  int local_failure =
      local_info == 0 ? context.size() : context.rank();
  int first_failing_rank = context.size();
  GEVD_MPI(MPI_Allreduce(&local_failure, &first_failing_rank, 1, MPI_INT,
                         MPI_MIN, context.communicator()));
  if (first_failing_rank == context.size()) {
    return;
  }

  int reported_info =
      context.rank() == first_failing_rank ? local_info : 0;
  GEVD_MPI(MPI_Bcast(&reported_info, 1, MPI_INT, first_failing_rank,
                     context.communicator()));
  throw std::runtime_error(std::string(stage) +
                           " returned numerical info " +
                           std::to_string(reported_info) + " on rank " +
                           std::to_string(first_failing_rank));
}

}  // namespace gevd4isc26::detail
