#include "common/distributed_problem.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace gevd4isc26::app {
namespace {

[[nodiscard]] std::uint64_t mixBits(std::uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

[[nodiscard]] double symmetricNoise(std::int64_t row,
                                    std::int64_t column) {
  const auto low = static_cast<std::uint64_t>(std::min(row, column));
  const auto high = static_cast<std::uint64_t>(std::max(row, column));
  const std::uint64_t bits =
      mixBits((low + 1) * 0x9e3779b97f4a7c15ULL ^
              (high + 1) * 0xd1b54a32d192ed03ULL);
  return static_cast<double>(bits >> 11) * 0x1.0p-53 - 0.5;
}

[[nodiscard]] std::vector<std::int64_t> sampleIndices(std::int64_t order,
                                                       int count) {
  if (count <= 0) {
    throw std::invalid_argument("sample count must be positive");
  }
  std::vector<std::int64_t> indices;
  indices.reserve(static_cast<std::size_t>(count));
  for (int sample = 0; sample < count; ++sample) {
    const std::int64_t index =
        count == 1
            ? order / 2
            : static_cast<std::int64_t>(
                  static_cast<long long>(sample) * (order - 1) /
                  (count - 1));
    if (indices.empty() || indices.back() != index) {
      indices.push_back(index);
    }
  }
  return indices;
}

[[nodiscard]] std::vector<double> fetchEigenvector(
    const DistributedMatrixView& Q,
    std::int64_t global_column,
    MPI_Comm communicator) {
  const MatrixDistribution& distribution = Q.distribution;
  std::vector<double> local(
      static_cast<std::size_t>(distribution.order()), 0.0);
  if (distribution.ownsGlobalColumn(global_column)) {
    const std::int64_t local_column =
        distribution.localColumn(global_column);
    for (std::int64_t local_row = 0;
         local_row < distribution.localRows(); ++local_row) {
      const std::int64_t global_row =
          distribution.globalRow(local_row);
      local[static_cast<std::size_t>(global_row)] =
          Q.data[local_row + local_column * Q.leading_dimension];
    }
  }

  std::vector<double> global(local.size(), 0.0);
  MPI_Allreduce(local.data(), global.data(), static_cast<int>(global.size()),
                MPI_DOUBLE, MPI_SUM, communicator);
  return global;
}

[[nodiscard]] std::vector<double> distributedMatvec(
    const DistributedMatrixView& matrix,
    const std::vector<double>& vector,
    MPI_Comm communicator) {
  const MatrixDistribution& distribution = matrix.distribution;
  std::vector<double> local(
      static_cast<std::size_t>(distribution.order()), 0.0);
  for (std::int64_t local_column = 0;
       local_column < distribution.localColumns(); ++local_column) {
    const std::int64_t global_column =
        distribution.globalColumn(local_column);
    const double x = vector[static_cast<std::size_t>(global_column)];
    for (std::int64_t local_row = 0;
         local_row < distribution.localRows(); ++local_row) {
      const std::int64_t global_row =
          distribution.globalRow(local_row);
      local[static_cast<std::size_t>(global_row)] +=
          matrix.data[local_row +
                      local_column * matrix.leading_dimension] *
          x;
    }
  }

  std::vector<double> global(local.size(), 0.0);
  MPI_Allreduce(local.data(), global.data(), static_cast<int>(global.size()),
                MPI_DOUBLE, MPI_SUM, communicator);
  return global;
}

[[nodiscard]] double dot(const std::vector<double>& left,
                         const std::vector<double>& right) {
  long double sum = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    sum += static_cast<long double>(left[index]) * right[index];
  }
  return static_cast<double>(sum);
}

}  // namespace

void generateDistributedProblem(DistributedMatrix& A, DistributedMatrix& B) {
  if (!A.distribution().sameLayoutAs(B.distribution()) ||
      A.leadingDimension() != B.leadingDimension()) {
    throw std::invalid_argument(
        "benchmark A and B must have identical layouts");
  }
  const MatrixDistribution& distribution = A.distribution();
  const std::int64_t order = distribution.order();
  const double inverse_sqrt_order =
      1.0 / std::sqrt(static_cast<double>(order));
  const double inverse_order = 1.0 / static_cast<double>(order);

  for (std::int64_t local_column = 0;
       local_column < distribution.localColumns(); ++local_column) {
    const std::int64_t column =
        distribution.globalColumn(local_column);
    for (std::int64_t local_row = 0;
         local_row < distribution.localRows(); ++local_row) {
      const std::int64_t row = distribution.globalRow(local_row);
      const std::size_t offset = static_cast<std::size_t>(
          local_row + local_column * A.leadingDimension());
      if (row == column) {
        A.values()[offset] =
            4.0 + static_cast<double>(row) * inverse_order;
        B.values()[offset] =
            2.0 + 0.25 * static_cast<double>(row) * inverse_order;
      } else {
        const double noise = symmetricNoise(row, column);
        A.values()[offset] = 0.2 * noise * inverse_sqrt_order;
        // The absolute row sum is below 0.025, making B strictly diagonally
        // dominant with a comfortable positive-definite margin.
        B.values()[offset] = 0.05 * noise * inverse_order;
      }
    }
  }
}

SampledValidation validateDistributedResult(
    const DistributedMatrixView& A,
    const DistributedMatrixView& B,
    const GeneralizedEigenResult& result,
    int sample_count,
    MPI_Comm communicator) {
  const std::int64_t order = A.distribution.order();
  const std::vector<std::int64_t> indices =
      sampleIndices(order, sample_count);
  std::vector<std::vector<double>> eigenvectors;
  std::vector<std::vector<double>> B_eigenvectors;
  eigenvectors.reserve(indices.size());
  B_eigenvectors.reserve(indices.size());

  SampledValidation validation;
  for (std::int64_t index : indices) {
    std::vector<double> q =
        fetchEigenvector(result.Q.view(), index, communicator);
    std::vector<double> Aq = distributedMatvec(A, q, communicator);
    std::vector<double> Bq = distributedMatvec(B, q, communicator);
    const double lambda =
        result.Lambda.diagonal()[static_cast<std::size_t>(index)];

    long double residual_squared = 0.0;
    long double scale_squared = 0.0;
    for (std::int64_t row = 0; row < order; ++row) {
      const double expected =
          lambda * Bq[static_cast<std::size_t>(row)];
      const double difference =
          Aq[static_cast<std::size_t>(row)] - expected;
      residual_squared +=
          static_cast<long double>(difference) * difference;
      scale_squared +=
          static_cast<long double>(expected) * expected;
    }
    validation.maximum_relative_residual = std::max(
        validation.maximum_relative_residual,
        std::sqrt(static_cast<double>(
            residual_squared /
            std::max<long double>(scale_squared, 1.0e-300L))));
    eigenvectors.push_back(std::move(q));
    B_eigenvectors.push_back(std::move(Bq));
  }

  for (std::size_t row = 0; row < eigenvectors.size(); ++row) {
    for (std::size_t column = 0; column < eigenvectors.size(); ++column) {
      const double expected = row == column ? 1.0 : 0.0;
      validation.maximum_B_orthogonality_error = std::max(
          validation.maximum_B_orthogonality_error,
          std::abs(dot(eigenvectors[row], B_eigenvectors[column]) -
                   expected));
    }
  }
  return validation;
}

}  // namespace gevd4isc26::app
