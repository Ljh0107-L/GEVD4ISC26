#include "gevd4isc26/matrix_io.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gevd4isc26 {
namespace {

constexpr int kScatterTag = 26001;
constexpr int kGatherTag = 26002;
constexpr std::size_t kMpiChunkElements = 1u << 26;

[[nodiscard]] std::size_t denseElementCount(std::int64_t order) {
  if (order <= 0) {
    throw std::invalid_argument("matrix order must be positive");
  }
  const auto n = static_cast<std::size_t>(order);
  if (n > std::numeric_limits<std::size_t>::max() / n) {
    throw std::overflow_error("dense matrix size overflows size_t");
  }
  return n * n;
}

void checkMpi(int status, const char* operation) {
  if (status == MPI_SUCCESS) {
    return;
  }
  char error_text[MPI_MAX_ERROR_STRING] = {};
  int length = 0;
  MPI_Error_string(status, error_text, &length);
  throw std::runtime_error(std::string(operation) + " failed: " +
                           std::string(error_text, length));
}

void sendLarge(const double* data,
               std::size_t count,
               int destination,
               int tag,
               MPI_Comm communicator) {
  std::size_t offset = 0;
  while (offset < count) {
    const int chunk = static_cast<int>(std::min(kMpiChunkElements, count - offset));
    checkMpi(MPI_Send(data + offset, chunk, MPI_DOUBLE, destination, tag, communicator),
             "MPI_Send matrix chunk");
    offset += static_cast<std::size_t>(chunk);
  }
}

void receiveLarge(double* data,
                  std::size_t count,
                  int source,
                  int tag,
                  MPI_Comm communicator) {
  std::size_t offset = 0;
  while (offset < count) {
    const int chunk = static_cast<int>(std::min(kMpiChunkElements, count - offset));
    checkMpi(MPI_Recv(data + offset, chunk, MPI_DOUBLE, source, tag, communicator,
                      MPI_STATUS_IGNORE),
             "MPI_Recv matrix chunk");
    offset += static_cast<std::size_t>(chunk);
  }
}

[[nodiscard]] std::vector<double> packShard(const std::vector<double>& full_matrix,
                                            const MatrixDistribution& distribution) {
  const std::int64_t local_rows = distribution.localRows();
  const std::int64_t local_columns = distribution.localColumns();
  std::vector<double> shard(static_cast<std::size_t>(local_rows * local_columns), 0.0);
  const std::int64_t n = distribution.order();

  for (std::int64_t local_column = 0; local_column < local_columns; ++local_column) {
    const std::int64_t global_column = distribution.globalColumn(local_column);
    for (std::int64_t local_row = 0; local_row < local_rows; ++local_row) {
      const std::int64_t global_row = distribution.globalRow(local_row);
      shard[static_cast<std::size_t>(local_row + local_column * local_rows)] =
          full_matrix[static_cast<std::size_t>(global_row + global_column * n)];
    }
  }
  return shard;
}

[[nodiscard]] std::vector<double> packWithoutPadding(const DistributedMatrixView& matrix) {
  const std::int64_t rows = matrix.distribution.localRows();
  const std::int64_t columns = matrix.distribution.localColumns();
  std::vector<double> packed(static_cast<std::size_t>(rows * columns), 0.0);
  for (std::int64_t column = 0; column < columns; ++column) {
    std::copy_n(matrix.data + column * matrix.leading_dimension, rows,
                packed.data() + column * rows);
  }
  return packed;
}

void unpackShard(const std::vector<double>& shard,
                 const MatrixDistribution& distribution,
                 std::vector<double>* full_matrix) {
  const std::int64_t local_rows = distribution.localRows();
  const std::int64_t local_columns = distribution.localColumns();
  const std::int64_t n = distribution.order();
  for (std::int64_t local_column = 0; local_column < local_columns; ++local_column) {
    const std::int64_t global_column = distribution.globalColumn(local_column);
    for (std::int64_t local_row = 0; local_row < local_rows; ++local_row) {
      const std::int64_t global_row = distribution.globalRow(local_row);
      (*full_matrix)[static_cast<std::size_t>(global_row + global_column * n)] =
          shard[static_cast<std::size_t>(local_row + local_column * local_rows)];
    }
  }
}

void validateCommunicator(const MatrixDistribution& distribution,
                          MPI_Comm communicator,
                          int root,
                          int* rank,
                          int* size) {
  checkMpi(MPI_Comm_rank(communicator, rank), "MPI_Comm_rank");
  checkMpi(MPI_Comm_size(communicator, size), "MPI_Comm_size");

  const long long local_metadata[8] = {
      static_cast<long long>(distribution.order()),
      static_cast<long long>(distribution.rowBlockSize()),
      static_cast<long long>(distribution.columnBlockSize()),
      static_cast<long long>(distribution.grid().rows),
      static_cast<long long>(distribution.grid().columns),
      static_cast<long long>(distribution.grid().source_row),
      static_cast<long long>(distribution.grid().source_column),
      static_cast<long long>(root)};
  long long minimum_metadata[8] = {};
  long long maximum_metadata[8] = {};
  checkMpi(MPI_Allreduce(local_metadata, minimum_metadata, 8, MPI_LONG_LONG_INT, MPI_MIN,
                         communicator),
           "MPI_Allreduce minimum matrix metadata");
  checkMpi(MPI_Allreduce(local_metadata, maximum_metadata, 8, MPI_LONG_LONG_INT, MPI_MAX,
                         communicator),
           "MPI_Allreduce maximum matrix metadata");
  bool metadata_matches = true;
  for (int index = 0; index < 8; ++index) {
    metadata_matches = metadata_matches && minimum_metadata[index] == maximum_metadata[index];
  }

  const bool locally_valid = metadata_matches && *size == distribution.grid().size() &&
                             *rank == distribution.rank() && root >= 0 && root < *size;
  int local_valid_integer = locally_valid ? 1 : 0;
  int all_valid = 0;
  checkMpi(MPI_Allreduce(&local_valid_integer, &all_valid, 1, MPI_INT, MPI_MIN, communicator),
           "MPI_Allreduce matrix I/O validation");
  if (all_valid == 0) {
    throw std::invalid_argument(
        "matrix I/O requires identical layouts, matching MPI ranks, and a valid root");
  }
}

void broadcastRootIoStatus(MPI_Comm communicator,
                           int root,
                           int rank,
                           const std::string& root_error) {
  int error_length = rank == root ? static_cast<int>(root_error.size()) : 0;
  checkMpi(MPI_Bcast(&error_length, 1, MPI_INT, root, communicator),
           "MPI_Bcast I/O status");
  if (error_length == 0) {
    return;
  }
  std::string error(static_cast<std::size_t>(error_length), '\0');
  if (rank == root) {
    error = root_error;
  }
  checkMpi(MPI_Bcast(error.data(), error_length, MPI_CHAR, root, communicator),
           "MPI_Bcast I/O error");
  throw std::runtime_error(error);
}

}  // namespace

DistributedMatrix scatterDenseMatrix(const std::vector<double>& full_matrix,
                                     const MatrixDistribution& local_distribution,
                                     MPI_Comm communicator,
                                     int root) {
  int rank = 0;
  int size = 1;
  validateCommunicator(local_distribution, communicator, root, &rank, &size);
  const std::size_t expected_elements = denseElementCount(local_distribution.order());

  std::string root_error;
  if (rank == root && full_matrix.size() != expected_elements) {
    root_error = "root's dense input matrix does not contain exactly n*n values";
  }
  broadcastRootIoStatus(communicator, root, rank, root_error);

  DistributedMatrix local_matrix(local_distribution);
  if (rank == root) {
    for (int destination = 0; destination < size; ++destination) {
      MatrixDistribution destination_distribution(
          local_distribution.order(), local_distribution.rowBlockSize(),
          local_distribution.columnBlockSize(), local_distribution.grid(), destination);
      std::vector<double> shard = packShard(full_matrix, destination_distribution);
      if (destination == root) {
        local_matrix.values() = std::move(shard);
      } else {
        sendLarge(shard.data(), shard.size(), destination, kScatterTag, communicator);
      }
    }
  } else {
    receiveLarge(local_matrix.data(), local_matrix.values().size(), root, kScatterTag,
                 communicator);
  }
  return local_matrix;
}

std::vector<double> gatherDenseMatrix(const DistributedMatrixView& local_matrix,
                                     MPI_Comm communicator,
                                     int root) {
  local_matrix.validate();
  int rank = 0;
  int size = 1;
  validateCommunicator(local_matrix.distribution, communicator, root, &rank, &size);
  std::vector<double> packed_local = packWithoutPadding(local_matrix);

  if (rank != root) {
    sendLarge(packed_local.data(), packed_local.size(), root, kGatherTag, communicator);
    return {};
  }

  std::vector<double> full_matrix(denseElementCount(local_matrix.distribution.order()), 0.0);
  for (int source = 0; source < size; ++source) {
    MatrixDistribution source_distribution(
        local_matrix.distribution.order(), local_matrix.distribution.rowBlockSize(),
        local_matrix.distribution.columnBlockSize(), local_matrix.distribution.grid(), source);
    std::vector<double> shard;
    if (source == root) {
      shard = packed_local;
    } else {
      shard.resize(static_cast<std::size_t>(source_distribution.localRows() *
                                            source_distribution.localColumns()));
      receiveLarge(shard.data(), shard.size(), source, kGatherTag, communicator);
    }
    unpackShard(shard, source_distribution, &full_matrix);
  }
  return full_matrix;
}

std::vector<double> readRawDenseMatrix(const std::string& path, std::int64_t order) {
  const std::size_t elements = denseElementCount(order);
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    throw std::overflow_error("dense matrix byte size overflows size_t");
  }
  const std::size_t expected_bytes = elements * sizeof(double);

  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open matrix file: " + path);
  }
  const std::streamoff file_bytes = input.tellg();
  if (file_bytes < 0 || static_cast<std::uint64_t>(file_bytes) != expected_bytes) {
    throw std::runtime_error("matrix file has the wrong size: " + path);
  }
  input.seekg(0);
  std::vector<double> matrix(elements);
  input.read(reinterpret_cast<char*>(matrix.data()), static_cast<std::streamsize>(expected_bytes));
  if (!input) {
    throw std::runtime_error("failed while reading matrix file: " + path);
  }
  return matrix;
}

void writeRawDenseMatrix(const std::string& path, const std::vector<double>& matrix) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create matrix file: " + path);
  }
  const auto bytes = static_cast<std::streamsize>(matrix.size() * sizeof(double));
  output.write(reinterpret_cast<const char*>(matrix.data()), bytes);
  if (!output) {
    throw std::runtime_error("failed while writing matrix file: " + path);
  }
}

void writeLambda(const std::string& path,
                 const DiagonalMatrix& lambda,
                 LambdaFileFormat format) {
  if (format == LambdaFileFormat::Diagonal) {
    writeRawDenseMatrix(path, lambda.diagonal());
    return;
  }

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create Lambda file: " + path);
  }
  const std::size_t n = static_cast<std::size_t>(lambda.order());
  std::vector<double> column(n, 0.0);
  for (std::size_t column_index = 0; column_index < n; ++column_index) {
    column[column_index] = lambda.diagonal()[column_index];
    output.write(reinterpret_cast<const char*>(column.data()),
                 static_cast<std::streamsize>(column.size() * sizeof(double)));
    column[column_index] = 0.0;
  }
  if (!output) {
    throw std::runtime_error("failed while writing Lambda file: " + path);
  }
}

}  // namespace gevd4isc26
