#pragma once

#include "runtime/error.hpp"

#include <cstdint>

namespace gevd4isc26::detail {

class SolverMatrixDescriptor {
public:
  SolverMatrixDescriptor(cusolverMpGrid_t grid,
                         std::int64_t order,
                         std::int64_t row_block_size,
                         std::int64_t column_block_size,
                         int source_row,
                         int source_column,
                         std::int64_t leading_dimension) {
    GEVD_CUSOLVER(cusolverMpCreateMatrixDesc(
        &descriptor_, grid, CUDA_R_64F, order, order, row_block_size,
        column_block_size, static_cast<std::uint32_t>(source_row),
        static_cast<std::uint32_t>(source_column), leading_dimension));
  }

  ~SolverMatrixDescriptor() {
    if (descriptor_ != nullptr) {
      cusolverMpDestroyMatrixDesc(descriptor_);
    }
  }

  SolverMatrixDescriptor(const SolverMatrixDescriptor&) = delete;
  SolverMatrixDescriptor& operator=(const SolverMatrixDescriptor&) = delete;

  [[nodiscard]] cusolverMpMatrixDescriptor_t get() const noexcept {
    return descriptor_;
  }

private:
  cusolverMpMatrixDescriptor_t descriptor_ = nullptr;
};

class BlasMatrixDescriptor {
public:
  BlasMatrixDescriptor(cublasMpGrid_t grid,
                       std::int64_t order,
                       std::int64_t row_block_size,
                       std::int64_t column_block_size,
                       int source_row,
                       int source_column,
                       std::int64_t leading_dimension) {
    GEVD_CUBLASMP(cublasMpMatrixDescriptorCreate(
        order, order, row_block_size, column_block_size, source_row,
        source_column, leading_dimension, CUDA_R_64F, grid, &descriptor_));
  }

  ~BlasMatrixDescriptor() {
    if (descriptor_ != nullptr) {
      cublasMpMatrixDescriptorDestroy(descriptor_);
    }
  }

  BlasMatrixDescriptor(const BlasMatrixDescriptor&) = delete;
  BlasMatrixDescriptor& operator=(const BlasMatrixDescriptor&) = delete;

  [[nodiscard]] cublasMpMatrixDescriptor_t get() const noexcept {
    return descriptor_;
  }

private:
  cublasMpMatrixDescriptor_t descriptor_ = nullptr;
};

}  // namespace gevd4isc26::detail
