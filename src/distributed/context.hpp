#pragma once

#include "gevd4isc26/distributed_matrix.hpp"
#include "runtime/gpu_resources.hpp"
#include "runtime/mpi_communicator.hpp"

#include <mpi.h>

namespace gevd4isc26::detail {

class DistributedGpuContext {
public:
  DistributedGpuContext(MPI_Comm communicator, ProcessGrid grid, int first_gpu);
  ~DistributedGpuContext();

  DistributedGpuContext(const DistributedGpuContext&) = delete;
  DistributedGpuContext& operator=(const DistributedGpuContext&) = delete;

  [[nodiscard]] MPI_Comm communicator() const noexcept {
    return communicator_.get();
  }
  [[nodiscard]] int rank() const noexcept { return communicator_.rank(); }
  [[nodiscard]] int size() const noexcept { return communicator_.size(); }
  [[nodiscard]] int processRow() const noexcept { return process_row_; }
  [[nodiscard]] int processColumn() const noexcept { return process_column_; }
  [[nodiscard]] int device() const noexcept { return device_; }
  [[nodiscard]] const ProcessGrid& gridShape() const noexcept { return grid_shape_; }
  [[nodiscard]] cudaStream_t stream() const noexcept { return stream_.get(); }
  [[nodiscard]] ncclComm_t nccl() const noexcept { return nccl_.get(); }
  [[nodiscard]] cusolverMpHandle_t solver() const noexcept {
    return solver_resources_.handle();
  }
  [[nodiscard]] cublasMpHandle_t blas() const noexcept {
    return blas_resources_.handle();
  }
  [[nodiscard]] cusolverMpGrid_t solverGrid() const noexcept {
    return solver_resources_.grid();
  }
  [[nodiscard]] cublasMpGrid_t blasGrid() const noexcept {
    return blas_resources_.grid();
  }

  void synchronize() const;

private:
  void initialize(MPI_Comm communicator, int first_gpu);

  ProcessGrid grid_shape_;
  MpiCommunicator communicator_;
  int process_row_ = 0;
  int process_column_ = 0;
  int device_ = -1;
  NcclCommunicator nccl_;
  CudaStream stream_;
  SolverMpResources solver_resources_;
  BlasMpResources blas_resources_;
};

}  // namespace gevd4isc26::detail
