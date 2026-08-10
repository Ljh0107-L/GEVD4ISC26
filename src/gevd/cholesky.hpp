#pragma once

#include "distributed/context.hpp"
#include "gevd/workspace.hpp"

#include <cstdint>

namespace gevd4isc26::detail {

// Stage 2 of GEVD: factor the symmetric-positive-definite metric matrix.
//
//   B = L L^T
class CholeskyStage {
public:
  CholeskyStage(const DistributedGpuContext& context,
                std::int64_t order,
                double* matrix_B,
                cusolverMpMatrixDescriptor_t descriptor_B);

  [[nodiscard]] WorkspaceSize queryWorkspace() const;

  void factorize(DeviceAllocation& device_workspace,
                 HostAllocation& host_workspace,
                 int* device_info) const;

private:
  const DistributedGpuContext& context_;
  std::int64_t order_;
  double* matrix_B_;
  cusolverMpMatrixDescriptor_t descriptor_B_;
};

}  // namespace gevd4isc26::detail
