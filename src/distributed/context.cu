#include "distributed/context.hpp"

#include "runtime/cpu_affinity.hpp"
#include "runtime/error.hpp"

#include <stdexcept>
#include <string>

namespace gevd4isc26::detail {

DistributedGpuContext::DistributedGpuContext(MPI_Comm communicator,
                                             ProcessGrid grid,
                                             int first_gpu)
    : grid_shape_(grid) {
  initialize(communicator, first_gpu);
}

DistributedGpuContext::~DistributedGpuContext() = default;

void DistributedGpuContext::initialize(MPI_Comm communicator, int first_gpu) {
  // cuBLASMp/NVSHMEM initialization may narrow the calling thread to the
  // GPU-local NUMA node. Preserve the rank-specific mask assigned by the
  // launcher across the complete resource initialization sequence.
  [[maybe_unused]] ScopedCpuAffinityRestore preserve_cpu_affinity;
  communicator_.duplicate(communicator);

  const int local_grid_values[4] = {grid_shape_.rows, grid_shape_.columns,
                                    grid_shape_.source_row, grid_shape_.source_column};
  int minimum_grid_values[4] = {};
  int maximum_grid_values[4] = {};
  GEVD_MPI(MPI_Allreduce(local_grid_values, minimum_grid_values, 4, MPI_INT, MPI_MIN,
                         communicator_.get()));
  GEVD_MPI(MPI_Allreduce(local_grid_values, maximum_grid_values, 4, MPI_INT, MPI_MAX,
                         communicator_.get()));
  for (int index = 0; index < 4; ++index) {
    if (minimum_grid_values[index] != maximum_grid_values[index]) {
      throw std::invalid_argument("all MPI ranks must use identical process-grid metadata");
    }
  }
  grid_shape_.validate();
  if (communicator_.size() != grid_shape_.size()) {
    throw std::invalid_argument("MPI communicator size must equal process_grid.rows * columns");
  }

  const auto coordinates = grid_shape_.coordinates(communicator_.rank());
  process_row_ = coordinates.first;
  process_column_ = coordinates.second;

  MPI_Comm node_communicator = MPI_COMM_NULL;
  GEVD_MPI(MPI_Comm_split_type(communicator_.get(), MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                               &node_communicator));
  int node_rank = 0;
  try {
    GEVD_MPI(MPI_Comm_rank(node_communicator, &node_rank));
    GEVD_MPI(MPI_Comm_free(&node_communicator));
  } catch (...) {
    if (node_communicator != MPI_COMM_NULL) {
      MPI_Comm_free(&node_communicator);
    }
    throw;
  }

  int visible_device_count = 0;
  GEVD_CUDA(cudaGetDeviceCount(&visible_device_count));
  if (visible_device_count <= 0) {
    throw std::runtime_error("no CUDA devices are visible to this MPI rank");
  }
  // Some launchers expose all node GPUs to every rank; others expose one GPU
  // per rank. The modulo mapping handles both cases exactly as the DFTB+ path.
  device_ = (first_gpu + node_rank) % visible_device_count;
  if (device_ < 0) {
    device_ += visible_device_count;
  }
  GEVD_CUDA(cudaSetDevice(device_));
  GEVD_CUDA(cudaFree(nullptr));

  nccl_.create(communicator_.get(), communicator_.rank(),
               communicator_.size());
  stream_.create(device_);
  solver_resources_.create(device_, stream_.get(), nccl_.get(),
                           grid_shape_.rows, grid_shape_.columns);
  blas_resources_.create(stream_.get(), nccl_.get(), grid_shape_.rows,
                         grid_shape_.columns);
}

void DistributedGpuContext::synchronize() const {
  GEVD_CUDA(cudaSetDevice(device_));
  GEVD_CUDA(cudaStreamSynchronize(stream_.get()));
}

}  // namespace gevd4isc26::detail
