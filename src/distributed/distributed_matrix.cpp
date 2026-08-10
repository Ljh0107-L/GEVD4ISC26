#include "gevd4isc26/distributed_matrix.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace gevd4isc26 {
namespace {

void requirePositive(std::int64_t value, const char* name) {
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
}

}  // namespace

int ProcessGrid::size() const {
  validate();
  return rows * columns;
}

std::pair<int, int> ProcessGrid::coordinates(int mpi_rank) const {
  validate();
  if (mpi_rank < 0 || mpi_rank >= rows * columns) {
    throw std::out_of_range("MPI rank is outside the process grid");
  }
  return {mpi_rank / columns, mpi_rank % columns};
}

int ProcessGrid::rank(int process_row, int process_column) const {
  validate();
  if (process_row < 0 || process_row >= rows || process_column < 0 ||
      process_column >= columns) {
    throw std::out_of_range("process coordinate is outside the process grid");
  }
  return process_row * columns + process_column;
}

void ProcessGrid::validate() const {
  if (rows <= 0 || columns <= 0) {
    throw std::invalid_argument("process grid dimensions must be positive");
  }
  if (source_row < 0 || source_row >= rows || source_column < 0 ||
      source_column >= columns) {
    throw std::invalid_argument("matrix source coordinate is outside the process grid");
  }
  if (rows > std::numeric_limits<int>::max() / columns) {
    throw std::overflow_error("process grid size overflows int");
  }
}

MatrixDistribution::MatrixDistribution(std::int64_t order,
                                       std::int64_t row_block_size,
                                       std::int64_t column_block_size,
                                       ProcessGrid grid,
                                       int rank)
    : order_(order),
      row_block_size_(row_block_size),
      column_block_size_(column_block_size),
      grid_(grid),
      rank_(rank),
      process_row_(0),
      process_column_(0),
      local_rows_(0),
      local_columns_(0) {
  requirePositive(order_, "matrix order");
  requirePositive(row_block_size_, "row block size");
  requirePositive(column_block_size_, "column block size");
  grid_.validate();

  const auto coordinates = grid_.coordinates(rank_);
  process_row_ = coordinates.first;
  process_column_ = coordinates.second;
  local_rows_ = localExtent(order_, row_block_size_, process_row_, grid_.source_row, grid_.rows);
  local_columns_ =
      localExtent(order_, column_block_size_, process_column_, grid_.source_column, grid_.columns);
}

std::int64_t MatrixDistribution::minimumLeadingDimension() const noexcept {
  return std::max<std::int64_t>(1, local_rows_);
}

std::size_t MatrixDistribution::storageSize(std::int64_t leading_dimension) const {
  if (leading_dimension < minimumLeadingDimension()) {
    throw std::invalid_argument("leading dimension is smaller than the local row count");
  }
  const auto ld = static_cast<std::uint64_t>(leading_dimension);
  const auto columns = static_cast<std::uint64_t>(local_columns_);
  if (columns != 0 && ld > std::numeric_limits<std::size_t>::max() / columns) {
    throw std::overflow_error("local matrix storage size overflows size_t");
  }
  return static_cast<std::size_t>(ld * columns);
}

bool MatrixDistribution::ownsGlobalRow(std::int64_t global_row) const {
  if (global_row < 0 || global_row >= order_) {
    return false;
  }
  return owner(global_row, row_block_size_, grid_.source_row, grid_.rows) == process_row_;
}

bool MatrixDistribution::ownsGlobalColumn(std::int64_t global_column) const {
  if (global_column < 0 || global_column >= order_) {
    return false;
  }
  return owner(global_column, column_block_size_, grid_.source_column, grid_.columns) ==
         process_column_;
}

std::int64_t MatrixDistribution::globalRow(std::int64_t local_row) const {
  if (local_row < 0 || local_row >= local_rows_) {
    throw std::out_of_range("local row is outside this rank's matrix shard");
  }
  return localToGlobal(local_row, row_block_size_, process_row_, grid_.source_row, grid_.rows);
}

std::int64_t MatrixDistribution::globalColumn(std::int64_t local_column) const {
  if (local_column < 0 || local_column >= local_columns_) {
    throw std::out_of_range("local column is outside this rank's matrix shard");
  }
  return localToGlobal(local_column, column_block_size_, process_column_, grid_.source_column,
                       grid_.columns);
}

std::int64_t MatrixDistribution::localRow(std::int64_t global_row) const {
  if (!ownsGlobalRow(global_row)) {
    return -1;
  }
  return globalToLocal(global_row, row_block_size_, process_row_, grid_.source_row, grid_.rows);
}

std::int64_t MatrixDistribution::localColumn(std::int64_t global_column) const {
  if (!ownsGlobalColumn(global_column)) {
    return -1;
  }
  return globalToLocal(global_column, column_block_size_, process_column_, grid_.source_column,
                       grid_.columns);
}

bool MatrixDistribution::sameLayoutAs(const MatrixDistribution& other) const noexcept {
  return order_ == other.order_ && row_block_size_ == other.row_block_size_ &&
         column_block_size_ == other.column_block_size_ && grid_.rows == other.grid_.rows &&
         grid_.columns == other.grid_.columns && grid_.source_row == other.grid_.source_row &&
         grid_.source_column == other.grid_.source_column && rank_ == other.rank_;
}

std::int64_t MatrixDistribution::localExtent(std::int64_t global_extent,
                                             std::int64_t block_size,
                                             int process_coordinate,
                                             int source_coordinate,
                                             int process_count) {
  requirePositive(global_extent, "global extent");
  requirePositive(block_size, "block size");
  if (process_count <= 0 || process_coordinate < 0 || process_coordinate >= process_count ||
      source_coordinate < 0 || source_coordinate >= process_count) {
    throw std::invalid_argument("invalid process coordinate in block-cyclic distribution");
  }

  std::int64_t result = 0;
  const std::int64_t block_count = (global_extent + block_size - 1) / block_size;
  for (std::int64_t block = 0; block < block_count; ++block) {
    if ((source_coordinate + block) % process_count == process_coordinate) {
      const std::int64_t block_begin = block * block_size;
      result += std::min(block_size, global_extent - block_begin);
    }
  }
  return result;
}

int MatrixDistribution::owner(std::int64_t global_index,
                              std::int64_t block_size,
                              int source_coordinate,
                              int process_count) {
  if (global_index < 0 || block_size <= 0 || process_count <= 0 || source_coordinate < 0 ||
      source_coordinate >= process_count) {
    throw std::invalid_argument("invalid index or distribution in owner()");
  }
  return static_cast<int>((source_coordinate + global_index / block_size) % process_count);
}

std::int64_t MatrixDistribution::localToGlobal(std::int64_t local_index,
                                               std::int64_t block_size,
                                               int process_coordinate,
                                               int source_coordinate,
                                               int process_count) {
  const std::int64_t local_block = local_index / block_size;
  const std::int64_t offset_in_block = local_index % block_size;
  const int first_owned_block =
      (process_coordinate - source_coordinate + process_count) % process_count;
  const std::int64_t global_block = local_block * process_count + first_owned_block;
  return global_block * block_size + offset_in_block;
}

std::int64_t MatrixDistribution::globalToLocal(std::int64_t global_index,
                                               std::int64_t block_size,
                                               int process_coordinate,
                                               int source_coordinate,
                                               int process_count) {
  const std::int64_t global_block = global_index / block_size;
  const std::int64_t offset_in_block = global_index % block_size;
  const int first_owned_block =
      (process_coordinate - source_coordinate + process_count) % process_count;
  return ((global_block - first_owned_block) / process_count) * block_size + offset_in_block;
}

void DistributedMatrixView::validate() const {
  if (data == nullptr) {
    throw std::invalid_argument("distributed matrix data pointer is null");
  }
  (void)distribution.storageSize(leading_dimension);
}

DistributedMatrix::DistributedMatrix(MatrixDistribution distribution)
    : DistributedMatrix(distribution, distribution.minimumLeadingDimension()) {}

DistributedMatrix::DistributedMatrix(MatrixDistribution distribution,
                                     std::int64_t leading_dimension)
    : distribution_(std::move(distribution)),
      leading_dimension_(leading_dimension),
      values_(distribution_.storageSize(leading_dimension_), 0.0) {}

DistributedMatrixView DistributedMatrix::view() const {
  return DistributedMatrixView{distribution_, data(), leading_dimension_};
}

}  // namespace gevd4isc26
