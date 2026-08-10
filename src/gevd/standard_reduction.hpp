#pragma once

#include "distributed/context.hpp"
#include "gevd/workspace.hpp"

#include <cstdint>

namespace gevd4isc26::detail {

// Stage 3 of GEVD: transform the generalized problem to standard form.
//
//   C = L^{-1} A L^{-T}
class StandardReductionStage {
public:
  StandardReductionStage(
      const DistributedGpuContext& context,
      std::int64_t order,
      double* matrix_A,
      double* cholesky_B,
      cusolverMpMatrixDescriptor_t descriptor_A,
      cusolverMpMatrixDescriptor_t descriptor_B);

  [[nodiscard]] WorkspaceSize queryWorkspace() const;

  void reduce(DeviceAllocation& device_workspace,
              HostAllocation& host_workspace,
              int* device_info) const;

private:
  const DistributedGpuContext& context_;
  std::int64_t order_;
  double* matrix_A_;
  double* cholesky_B_;
  cusolverMpMatrixDescriptor_t descriptor_A_;
  cusolverMpMatrixDescriptor_t descriptor_B_;
};

}  // namespace gevd4isc26::detail
