#include "gevd/cholesky.hpp"

#include "runtime/error.hpp"

#include <cuda_runtime.h>
#include <cusolverMp.h>

namespace gevd4isc26::detail {

CholeskyStage::CholeskyStage(
    const DistributedGpuContext& context,
    std::int64_t order,
    double* matrix_B,
    cusolverMpMatrixDescriptor_t descriptor_B)
    : context_(context),
      order_(order),
      matrix_B_(matrix_B),
      descriptor_B_(descriptor_B) {}

WorkspaceSize CholeskyStage::queryWorkspace() const {
  WorkspaceSize workspace;
  GEVD_CUSOLVER(cusolverMpPotrf_bufferSize(
      context_.solver(), CUBLAS_FILL_MODE_LOWER, order_, matrix_B_, 1, 1,
      descriptor_B_, CUDA_R_64F, &workspace.device_bytes,
      &workspace.host_bytes));
  return workspace;
}

void CholeskyStage::factorize(DeviceAllocation& device_workspace,
                              HostAllocation& host_workspace,
                              int* device_info) const {
  GEVD_CUDA(cudaMemsetAsync(device_info, 0, sizeof(int), context_.stream()));
  GEVD_CUSOLVER(cusolverMpPotrf(
      context_.solver(), CUBLAS_FILL_MODE_LOWER, order_, matrix_B_, 1, 1,
      descriptor_B_, CUDA_R_64F, device_workspace.data(),
      device_workspace.bytes(), host_workspace.data(), host_workspace.bytes(),
      device_info));
}

}  // namespace gevd4isc26::detail
