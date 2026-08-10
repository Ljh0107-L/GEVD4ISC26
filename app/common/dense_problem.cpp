#include "common/dense_problem.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace gevd4isc26::app {
namespace {

[[nodiscard]] double& at(DenseMatrix& matrix,
                         std::int64_t order,
                         std::int64_t row,
                         std::int64_t column) {
  return matrix[static_cast<std::size_t>(row + column * order)];
}

[[nodiscard]] double at(const DenseMatrix& matrix,
                        std::int64_t order,
                        std::int64_t row,
                        std::int64_t column) {
  return matrix[static_cast<std::size_t>(row + column * order)];
}

[[nodiscard]] DenseMatrix multiply(const DenseMatrix& left,
                                   const DenseMatrix& right,
                                   std::int64_t order) {
  DenseMatrix product(static_cast<std::size_t>(order * order), 0.0);
  for (std::int64_t column = 0; column < order; ++column) {
    for (std::int64_t inner = 0; inner < order; ++inner) {
      const double right_value = at(right, order, inner, column);
      for (std::int64_t row = 0; row < order; ++row) {
        at(product, order, row, column) +=
            at(left, order, row, inner) * right_value;
      }
    }
  }
  return product;
}

[[nodiscard]] DenseMatrix transpose(const DenseMatrix& matrix,
                                    std::int64_t order) {
  DenseMatrix result(static_cast<std::size_t>(order * order));
  for (std::int64_t column = 0; column < order; ++column) {
    for (std::int64_t row = 0; row < order; ++row) {
      at(result, order, row, column) = at(matrix, order, column, row);
    }
  }
  return result;
}

[[nodiscard]] double frobeniusNorm(const DenseMatrix& matrix) {
  double sum = 0.0;
  for (double value : matrix) {
    sum += value * value;
  }
  return std::sqrt(sum);
}

}  // namespace

KnownProblem makeKnownProblem(std::int64_t order) {
  DenseMatrix L(static_cast<std::size_t>(order * order), 0.0);
  for (std::int64_t row = 0; row < order; ++row) {
    at(L, order, row, row) = 1.5 + 0.001 * static_cast<double>(row);
    for (std::int64_t column = std::max<std::int64_t>(0, row - 3);
         column < row; ++column) {
      at(L, order, row, column) =
          0.01 / static_cast<double>(row - column);
    }
  }

  DenseMatrix U(static_cast<std::size_t>(order * order), 0.0);
  const double pi = std::acos(-1.0);
  const double first_column_scale =
      1.0 / std::sqrt(static_cast<double>(order));
  const double other_column_scale =
      std::sqrt(2.0 / static_cast<double>(order));
  for (std::int64_t column = 0; column < order; ++column) {
    const double scale =
        column == 0 ? first_column_scale : other_column_scale;
    for (std::int64_t row = 0; row < order; ++row) {
      at(U, order, row, column) =
          scale * std::cos(pi * (static_cast<double>(row) + 0.5) *
                           static_cast<double>(column) /
                           static_cast<double>(order));
    }
  }

  std::vector<double> eigenvalues(static_cast<std::size_t>(order));
  DenseMatrix U_lambda = U;
  for (std::int64_t column = 0; column < order; ++column) {
    eigenvalues[static_cast<std::size_t>(column)] =
        1.0 + 0.01 * static_cast<double>(column);
    for (std::int64_t row = 0; row < order; ++row) {
      at(U_lambda, order, row, column) *=
          eigenvalues[static_cast<std::size_t>(column)];
    }
  }

  const DenseMatrix B = multiply(L, transpose(L, order), order);
  const DenseMatrix middle =
      multiply(U_lambda, transpose(U, order), order);
  const DenseMatrix A =
      multiply(multiply(L, middle, order), transpose(L, order), order);
  return {A, B, eigenvalues};
}

DenseValidation validateKnownProblem(
    const KnownProblem& problem,
    const DenseMatrix& Q,
    const std::vector<double>& eigenvalues,
    std::int64_t order) {
  DenseMatrix AQ = multiply(problem.A, Q, order);
  DenseMatrix BQ = multiply(problem.B, Q, order);
  DenseMatrix BQ_lambda = BQ;
  for (std::int64_t column = 0; column < order; ++column) {
    for (std::int64_t row = 0; row < order; ++row) {
      at(BQ_lambda, order, row, column) *=
          eigenvalues[static_cast<std::size_t>(column)];
      at(AQ, order, row, column) -=
          at(BQ_lambda, order, row, column);
    }
  }

  DenseValidation validation;
  validation.relative_residual =
      frobeniusNorm(AQ) / std::max(1.0, frobeniusNorm(BQ_lambda));

  DenseMatrix gram = multiply(transpose(Q, order), BQ, order);
  for (std::int64_t index = 0; index < order; ++index) {
    at(gram, order, index, index) -= 1.0;
  }
  validation.B_orthogonality_error =
      frobeniusNorm(gram) / std::sqrt(static_cast<double>(order));

  for (std::int64_t index = 0; index < order; ++index) {
    validation.eigenvalue_error = std::max(
        validation.eigenvalue_error,
        std::abs(eigenvalues[static_cast<std::size_t>(index)] -
                 problem.eigenvalues[static_cast<std::size_t>(index)]));
  }
  return validation;
}

}  // namespace gevd4isc26::app
