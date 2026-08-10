#include "gevd/workspace.hpp"

#include "runtime/error.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>

namespace gevd4isc26::detail {
namespace {

[[nodiscard]] std::size_t matrixBytes(
    const MatrixDistribution& distribution,
    std::int64_t leading_dimension) {
  const std::size_t elements = distribution.storageSize(leading_dimension);
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    throw std::overflow_error("local matrix byte size overflows size_t");
  }
  return elements * sizeof(double);
}

}  // namespace

GevdWorkspace::GevdWorkspace(
    const DistributedGpuContext& context,
    const DistributedMatrixView& A,
    const DistributedMatrixView& B,
    bool prepare_inverse_concurrently,
    DeviceAllocation& retained_eigenvectors)
    : context_(context),
      distribution_(A.distribution),
      order_(distribution_.order()),
      leading_dimension_(A.leading_dimension),
      local_matrix_bytes_(matrixBytes(distribution_, leading_dimension_)),
      output_eigenvectors_(distribution_, leading_dimension_),
      eigenvalues_(static_cast<std::size_t>(order_), 0.0),
      standard_matrix_(local_matrix_bytes_),
      generalized_eigenvectors_(
          prepare_inverse_concurrently ? local_matrix_bytes_ : 0),
      numerical_info_(sizeof(int)),
      solver_A_(
          context_.solverGrid(), order_, distribution_.rowBlockSize(),
          distribution_.columnBlockSize(), distribution_.grid().source_row,
          distribution_.grid().source_column, A.leading_dimension),
      solver_B_(
          context_.solverGrid(), order_, distribution_.rowBlockSize(),
          distribution_.columnBlockSize(), distribution_.grid().source_row,
          distribution_.grid().source_column, B.leading_dimension),
      blas_B_(
          context_.blasGrid(), order_, distribution_.rowBlockSize(),
          distribution_.columnBlockSize(), distribution_.grid().source_row,
          distribution_.grid().source_column, B.leading_dimension),
      blas_Q_(
          context_.blasGrid(), order_, distribution_.rowBlockSize(),
          distribution_.columnBlockSize(), distribution_.grid().source_row,
          distribution_.grid().source_column, A.leading_dimension) {
  if (retained_eigenvectors.bytes() != local_matrix_bytes_) {
    retained_eigenvectors.allocate(local_matrix_bytes_);
  }
  standard_eigenvectors_ = retained_eigenvectors.as<double>();
}

void GevdWorkspace::allocateNumericalWorkspace(const WorkspaceSize& size) {
  device_workspace_.allocate(size.device_bytes);
  host_workspace_.allocate(size.host_bytes);
}

void GevdWorkspace::uploadInputs(const DistributedMatrixView& A,
                                 const DistributedMatrixView& B,
                                 double* cholesky_B,
                                 bool reuse_cholesky) {
  GEVD_CUDA(cudaMemcpyAsync(standard_matrix_.data(), A.data,
                            local_matrix_bytes_, cudaMemcpyHostToDevice,
                            context_.stream()));
  if (!reuse_cholesky) {
    GEVD_CUDA(cudaMemcpyAsync(cholesky_B, B.data, local_matrix_bytes_,
                              cudaMemcpyHostToDevice, context_.stream()));
  }
  GEVD_CUDA(cudaMemsetAsync(standard_eigenvectors_, 0, local_matrix_bytes_,
                            context_.stream()));
  if (generalized_eigenvectors_.bytes() != 0) {
    GEVD_CUDA(cudaMemsetAsync(generalized_eigenvectors_.data(), 0,
                              local_matrix_bytes_, context_.stream()));
  }
  context_.synchronize();
}

void GevdWorkspace::downloadEigenvectors(const double* device_eigenvectors) {
  GEVD_CUDA(cudaMemcpyAsync(output_eigenvectors_.data(), device_eigenvectors,
                            local_matrix_bytes_, cudaMemcpyDeviceToHost,
                            context_.stream()));
  context_.synchronize();
}

}  // namespace gevd4isc26::detail
