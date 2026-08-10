#pragma once

#include "distributed/context.hpp"
#include "gevd4isc26/distributed_matrix.hpp"
#include "runtime/resources.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace gevd4isc26::detail {

// A numerical stage may need both GPU workspace and page-able host workspace.
// The GEVD pipeline reuses one allocation sized to the largest stage.
struct WorkspaceSize {
  std::size_t device_bytes = 0;
  std::size_t host_bytes = 0;

  void include(std::size_t device, std::size_t host) noexcept {
    device_bytes = std::max(device_bytes, device);
    host_bytes = std::max(host_bytes, host);
  }

  void include(const WorkspaceSize& other) noexcept {
    include(other.device_bytes, other.host_bytes);
  }
};

// Owns all shape-dependent, per-call storage for one complete generalized
// eigensolve. B-dependent factors and the reusable Y buffer remain owned by
// GeneralizedEigensolver so they can survive across calls.
class GevdWorkspace {
public:
  GevdWorkspace(const DistributedGpuContext& context,
                const DistributedMatrixView& A,
                const DistributedMatrixView& B,
                bool prepare_inverse_concurrently,
                DeviceAllocation& retained_eigenvectors);

  void allocateNumericalWorkspace(const WorkspaceSize& size);

  void uploadInputs(const DistributedMatrixView& A,
                    const DistributedMatrixView& B,
                    double* cholesky_B,
                    bool reuse_cholesky);

  void downloadEigenvectors(const double* device_eigenvectors);

  [[nodiscard]] const MatrixDistribution& distribution() const noexcept {
    return distribution_;
  }
  [[nodiscard]] std::int64_t order() const noexcept { return order_; }
  [[nodiscard]] std::int64_t leadingDimension() const noexcept {
    return leading_dimension_;
  }
  [[nodiscard]] std::size_t localMatrixBytes() const noexcept {
    return local_matrix_bytes_;
  }

  [[nodiscard]] DeviceAllocation& standardMatrix() noexcept {
    return standard_matrix_;
  }
  [[nodiscard]] double* standardEigenvectors() noexcept {
    return standard_eigenvectors_;
  }
  [[nodiscard]] double* generalizedEigenvectors() noexcept {
    return generalized_eigenvectors_.as<double>();
  }
  [[nodiscard]] int* numericalInfo() noexcept {
    return numerical_info_.as<int>();
  }

  [[nodiscard]] DeviceAllocation& deviceWorkspace() noexcept {
    return device_workspace_;
  }
  [[nodiscard]] HostAllocation& hostWorkspace() noexcept {
    return host_workspace_;
  }

  [[nodiscard]] cusolverMpMatrixDescriptor_t solverDescriptorA() const noexcept {
    return solver_A_.get();
  }
  [[nodiscard]] cusolverMpMatrixDescriptor_t solverDescriptorB() const noexcept {
    return solver_B_.get();
  }
  [[nodiscard]] cublasMpMatrixDescriptor_t blasDescriptorB() const noexcept {
    return blas_B_.get();
  }
  [[nodiscard]] cublasMpMatrixDescriptor_t blasDescriptorQ() const noexcept {
    return blas_Q_.get();
  }

  [[nodiscard]] std::vector<double>& eigenvalues() noexcept {
    return eigenvalues_;
  }
  [[nodiscard]] DistributedMatrix&& releaseEigenvectors() noexcept {
    return std::move(output_eigenvectors_);
  }
  [[nodiscard]] std::vector<double>&& releaseEigenvalues() noexcept {
    return std::move(eigenvalues_);
  }

private:
  const DistributedGpuContext& context_;
  MatrixDistribution distribution_;
  std::int64_t order_;
  std::int64_t leading_dimension_;
  std::size_t local_matrix_bytes_;

  DistributedMatrix output_eigenvectors_;
  std::vector<double> eigenvalues_;
  DeviceAllocation standard_matrix_;
  double* standard_eigenvectors_ = nullptr;
  DeviceAllocation generalized_eigenvectors_;
  DeviceAllocation numerical_info_;

  SolverMatrixDescriptor solver_A_;
  SolverMatrixDescriptor solver_B_;
  BlasMatrixDescriptor blas_B_;
  BlasMatrixDescriptor blas_Q_;

  DeviceAllocation device_workspace_;
  HostAllocation host_workspace_;
};

}  // namespace gevd4isc26::detail
