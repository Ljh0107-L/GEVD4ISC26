#pragma once

#include <cstdint>
#include <vector>

namespace gevd4isc26::app {

using DenseMatrix = std::vector<double>;

struct KnownProblem {
  DenseMatrix A;
  DenseMatrix B;
  std::vector<double> eigenvalues;
};

struct DenseValidation {
  double eigenvalue_error = 0.0;
  double relative_residual = 0.0;
  double B_orthogonality_error = 0.0;

  [[nodiscard]] bool passed(double tolerance = 1.0e-8) const noexcept {
    return eigenvalue_error < tolerance &&
           relative_residual < tolerance &&
           B_orthogonality_error < tolerance;
  }
};

// Builds B = L L^T and A = L U Lambda U^T L^T. The exact generalized
// eigenvectors are Q = L^{-T} U and Q^T B Q = I.
[[nodiscard]] KnownProblem makeKnownProblem(std::int64_t order);

[[nodiscard]] DenseValidation validateKnownProblem(
    const KnownProblem& problem,
    const DenseMatrix& Q,
    const std::vector<double>& eigenvalues,
    std::int64_t order);

}  // namespace gevd4isc26::app
