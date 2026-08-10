#pragma once

#include "distributed/context.hpp"
#include "gevd/workspace.hpp"
#include "runtime/resources.hpp"
#include "gevd4isc26/distributed_matrix.hpp"

#include <cstddef>
#include <cstdint>

namespace gevd4isc26::detail {

WorkspaceSize queryDirectBacktransformWorkspace(
    const DistributedGpuContext& context,
    std::int64_t order,
    double* cholesky_factor,
    double* eigenvectors,
    cublasMpMatrixDescriptor_t cholesky_descriptor,
    cublasMpMatrixDescriptor_t eigenvector_descriptor);

WorkspaceSize queryOverlapBacktransformWorkspace(
    const DistributedGpuContext& context,
    std::int64_t order,
    double* cholesky_factor,
    double* inverse_transpose,
    double* eigenvectors,
    double* output,
    cublasMpMatrixDescriptor_t cholesky_descriptor,
    cublasMpMatrixDescriptor_t matrix_descriptor,
    bool inverse_is_cached);

void backtransformWithTriangularSolve(
    const DistributedGpuContext& context,
    std::int64_t order,
    double* cholesky_factor,
    double* eigenvectors,
    cublasMpMatrixDescriptor_t cholesky_descriptor,
    cublasMpMatrixDescriptor_t eigenvector_descriptor,
    DeviceAllocation& device_workspace,
    HostAllocation& host_workspace);

// The standard EVD stage invokes callback() after applying its reflectors and
// before joining the CPU divide-and-conquer eigensolve.  The callback enqueues
// L^{-T}, overlapping GEVD work from two otherwise independent stages.
class OverlapInverseBacktransform {
public:
  OverlapInverseBacktransform(
      const DistributedGpuContext& context,
      const MatrixDistribution& distribution,
      std::int64_t leading_dimension,
      double* cholesky_factor,
      double* inverse_transpose,
      cublasMpMatrixDescriptor_t cholesky_descriptor,
      cublasMpMatrixDescriptor_t matrix_descriptor,
      DeviceAllocation& device_workspace,
      HostAllocation& host_workspace,
      bool inverse_is_cached);
  ~OverlapInverseBacktransform();

  [[nodiscard]] int (*callback() const noexcept)(void*);
  [[nodiscard]] void* callbackData() noexcept { return this; }

  void wait();
  void multiply(double* eigenvectors, double* output);
  [[nodiscard]] bool inverseWasComputed() const noexcept;
  [[nodiscard]] double inversePreparationSeconds() const noexcept;

private:
  static int invoke(void* user_data) noexcept;
  int launch() noexcept;

  const DistributedGpuContext& context_;
  const MatrixDistribution& distribution_;
  std::int64_t leading_dimension_;
  double* cholesky_factor_;
  double* inverse_transpose_;
  cublasMpMatrixDescriptor_t cholesky_descriptor_;
  cublasMpMatrixDescriptor_t matrix_descriptor_;
  DeviceAllocation& device_workspace_;
  HostAllocation& host_workspace_;
  bool inverse_is_cached_;
  bool launched_ = false;
  float inverse_milliseconds_ = 0.0F;
  cudaEvent_t inverse_start_ = nullptr;
  cudaEvent_t inverse_end_ = nullptr;
  cudaError_t cuda_status_ = cudaSuccess;
  cublasMpStatus_t cublas_status_ = CUBLASMP_STATUS_SUCCESS;
};

}  // namespace gevd4isc26::detail
