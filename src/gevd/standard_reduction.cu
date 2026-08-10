#include "gevd/standard_reduction.hpp"

#include "runtime/error.hpp"

#include <cuda_runtime.h>
#include <cusolverMp.h>

namespace gevd4isc26::detail {

StandardReductionStage::StandardReductionStage(
    const DistributedGpuContext& context,
    std::int64_t order,
    double* matrix_A,
    double* cholesky_B,
    cusolverMpMatrixDescriptor_t descriptor_A,
    cusolverMpMatrixDescriptor_t descriptor_B)
    : context_(context),
      order_(order),
      matrix_A_(matrix_A),
      cholesky_B_(cholesky_B),
      descriptor_A_(descriptor_A),
      descriptor_B_(descriptor_B) {}

WorkspaceSize StandardReductionStage::queryWorkspace() const {
  WorkspaceSize workspace;
  GEVD_CUSOLVER(cusolverMpSygst_bufferSize(
      context_.solver(), CUSOLVER_EIG_TYPE_1, CUBLAS_FILL_MODE_LOWER, order_,
      1, 1, descriptor_A_, 1, 1, descriptor_B_, CUDA_R_64F,
      &workspace.device_bytes, &workspace.host_bytes));
  return workspace;
}

void StandardReductionStage::reduce(DeviceAllocation& device_workspace,
                                    HostAllocation& host_workspace,
                                    int* device_info) const {
  GEVD_CUDA(cudaMemsetAsync(device_info, 0, sizeof(int), context_.stream()));
  GEVD_CUSOLVER(cusolverMpSygst(
      context_.solver(), CUSOLVER_EIG_TYPE_1, CUBLAS_FILL_MODE_LOWER, order_,
      matrix_A_, 1, 1, descriptor_A_, cholesky_B_, 1, 1, descriptor_B_,
      CUDA_R_64F, device_workspace.data(), device_workspace.bytes(),
      host_workspace.data(), host_workspace.bytes(), device_info));
}

}  // namespace gevd4isc26::detail
