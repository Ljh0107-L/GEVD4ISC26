#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace gevd4isc26 {

// MPI ranks are mapped to this grid in row-major order:
// rank = process_row * columns + process_column.
struct ProcessGrid {
  int rows = 1;
  int columns = 1;
  int source_row = 0;
  int source_column = 0;

  [[nodiscard]] int size() const;
  [[nodiscard]] std::pair<int, int> coordinates(int rank) const;
  [[nodiscard]] int rank(int process_row, int process_column) const;
  void validate() const;
};

// Describes one rank's part of a square, column-major, two-dimensional
// block-cyclic matrix. The layout is compatible with BLACS/ScaLAPACK and
// with the cuSOLVERMp descriptors used by DFTB+.
class MatrixDistribution {
public:
  MatrixDistribution(std::int64_t order,
                     std::int64_t row_block_size,
                     std::int64_t column_block_size,
                     ProcessGrid grid,
                     int rank);

  [[nodiscard]] std::int64_t order() const noexcept { return order_; }
  [[nodiscard]] std::int64_t rowBlockSize() const noexcept { return row_block_size_; }
  [[nodiscard]] std::int64_t columnBlockSize() const noexcept {
    return column_block_size_;
  }
  [[nodiscard]] const ProcessGrid& grid() const noexcept { return grid_; }
  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int processRow() const noexcept { return process_row_; }
  [[nodiscard]] int processColumn() const noexcept { return process_column_; }
  [[nodiscard]] std::int64_t localRows() const noexcept { return local_rows_; }
  [[nodiscard]] std::int64_t localColumns() const noexcept { return local_columns_; }
  [[nodiscard]] std::int64_t minimumLeadingDimension() const noexcept;
  [[nodiscard]] std::size_t storageSize(std::int64_t leading_dimension) const;

  [[nodiscard]] bool ownsGlobalRow(std::int64_t global_row) const;
  [[nodiscard]] bool ownsGlobalColumn(std::int64_t global_column) const;
  [[nodiscard]] std::int64_t globalRow(std::int64_t local_row) const;
  [[nodiscard]] std::int64_t globalColumn(std::int64_t local_column) const;
  [[nodiscard]] std::int64_t localRow(std::int64_t global_row) const;
  [[nodiscard]] std::int64_t localColumn(std::int64_t global_column) const;

  [[nodiscard]] bool sameLayoutAs(const MatrixDistribution& other) const noexcept;

  // Public helpers used by matrix I/O and external distributed-data adapters.
  [[nodiscard]] static std::int64_t localExtent(std::int64_t global_extent,
                                                std::int64_t block_size,
                                                int process_coordinate,
                                                int source_coordinate,
                                                int process_count);
  [[nodiscard]] static int owner(std::int64_t global_index,
                                 std::int64_t block_size,
                                 int source_coordinate,
                                 int process_count);

private:
  [[nodiscard]] static std::int64_t localToGlobal(std::int64_t local_index,
                                                  std::int64_t block_size,
                                                  int process_coordinate,
                                                  int source_coordinate,
                                                  int process_count);
  [[nodiscard]] static std::int64_t globalToLocal(std::int64_t global_index,
                                                  std::int64_t block_size,
                                                  int process_coordinate,
                                                  int source_coordinate,
                                                  int process_count);

  std::int64_t order_;
  std::int64_t row_block_size_;
  std::int64_t column_block_size_;
  ProcessGrid grid_;
  int rank_;
  int process_row_;
  int process_column_;
  std::int64_t local_rows_;
  std::int64_t local_columns_;
};

// A non-owning view of local host memory. data is column-major and may have
// padding between columns through leading_dimension.
struct DistributedMatrixView {
  MatrixDistribution distribution;
  const double* data = nullptr;
  std::int64_t leading_dimension = 0;

  void validate() const;
};

// Owning host representation of one rank's local matrix shard.
class DistributedMatrix {
public:
  explicit DistributedMatrix(MatrixDistribution distribution);
  DistributedMatrix(MatrixDistribution distribution, std::int64_t leading_dimension);

  [[nodiscard]] const MatrixDistribution& distribution() const noexcept {
    return distribution_;
  }
  [[nodiscard]] std::int64_t leadingDimension() const noexcept { return leading_dimension_; }
  [[nodiscard]] double* data() noexcept { return values_.data(); }
  [[nodiscard]] const double* data() const noexcept { return values_.data(); }
  [[nodiscard]] std::vector<double>& values() noexcept { return values_; }
  [[nodiscard]] const std::vector<double>& values() const noexcept { return values_; }
  [[nodiscard]] DistributedMatrixView view() const;

private:
  MatrixDistribution distribution_;
  std::int64_t leading_dimension_;
  std::vector<double> values_;
};

}  // namespace gevd4isc26
