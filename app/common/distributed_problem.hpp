#pragma once

#include "gevd4isc26/distributed_matrix.hpp"
#include "gevd4isc26/solver.hpp"

#include <mpi.h>

namespace gevd4isc26::app {

struct SampledValidation {
  double maximum_relative_residual = 0.0;
  double maximum_B_orthogonality_error = 0.0;

  [[nodiscard]] bool passed(double tolerance = 1.0e-8) const noexcept {
    return maximum_relative_residual < tolerance &&
           maximum_B_orthogonality_error < tolerance;
  }
};

// Fills a reproducible dense symmetric A and a strictly diagonally-dominant,
// symmetric-positive-definite B directly in each rank's block-cyclic shard.
void generateDistributedProblem(DistributedMatrix& A, DistributedMatrix& B);

// Validates sampled columns without gathering full n-by-n matrices. Matrix
// vector products are accumulated collectively over the 2-D process grid.
[[nodiscard]] SampledValidation validateDistributedResult(
    const DistributedMatrixView& A,
    const DistributedMatrixView& B,
    const GeneralizedEigenResult& result,
    int sample_count,
    MPI_Comm communicator);

}  // namespace gevd4isc26::app
