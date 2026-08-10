#pragma once

#include "distributed/context.hpp"
#include "runtime/resources.hpp"
#include "gevd4isc26/distributed_matrix.hpp"

#include <cstddef>
#include <cstdint>

namespace gevd4isc26::detail {

struct FactorReuse {
  bool cholesky = false;
  bool inverse = false;
};

// Owns the two B-dependent GPU matrices that can be reused across solve calls:
// the Cholesky factor L and, for the optimized backtransform, L^{-T}.
class FactorizationCache {
public:
  FactorReuse prepare(const DistributedGpuContext& context,
                      const DistributedMatrixView& B,
                      std::int64_t inverse_leading_dimension,
                      bool cache_cholesky,
                      bool cache_inverse,
                      bool need_inverse);

  [[nodiscard]] double* cholesky() noexcept { return cholesky_.as<double>(); }
  [[nodiscard]] double* inverse() noexcept { return inverse_.as<double>(); }

  void markCholeskyValid(bool cache_enabled) noexcept;
  void markInverseValid(bool cache_enabled) noexcept;

private:
  struct MatrixKey {
    std::int64_t order = 0;
    std::int64_t row_block_size = 0;
    std::int64_t column_block_size = 0;
    std::int64_t leading_dimension = 0;
    std::int64_t local_columns = 0;
    int grid_rows = 0;
    int grid_columns = 0;
    int source_row = 0;
    int source_column = 0;
    int process_row = 0;
    int process_column = 0;
    std::size_t bytes = 0;
    std::uint64_t fingerprint = 0;

    [[nodiscard]] bool operator==(const MatrixKey& other) const noexcept {
      return order == other.order &&
             row_block_size == other.row_block_size &&
             column_block_size == other.column_block_size &&
             leading_dimension == other.leading_dimension &&
             local_columns == other.local_columns &&
             grid_rows == other.grid_rows &&
             grid_columns == other.grid_columns &&
             source_row == other.source_row &&
             source_column == other.source_column &&
             process_row == other.process_row &&
             process_column == other.process_column && bytes == other.bytes &&
             fingerprint == other.fingerprint;
    }
  };

  DeviceAllocation cholesky_;
  DeviceAllocation inverse_;
  MatrixKey cholesky_key_;
  MatrixKey inverse_basis_key_;
  std::int64_t inverse_leading_dimension_ = 0;
  std::size_t inverse_bytes_ = 0;
  bool cholesky_key_set_ = false;
  bool inverse_key_set_ = false;
  bool cholesky_valid_ = false;
  bool inverse_valid_ = false;
};

}  // namespace gevd4isc26::detail
