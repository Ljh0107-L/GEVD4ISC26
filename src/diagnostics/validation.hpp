#pragma once

#include "distributed/context.hpp"
#include "gevd4isc26/distributed_matrix.hpp"
#include "gevd4isc26/solver.hpp"

namespace gevd4isc26::detail {

// Validates both the local matrix views and the collective contract shared by
// all MPI ranks.  Every rank either accepts the same problem or receives the
// error reported by the first failing rank.
void validateGeneralizedEigenproblem(const DistributedGpuContext& context,
                                     const DistributedMatrixView& A,
                                     const DistributedMatrixView& B,
                                     const SolverOptions& options);

}  // namespace gevd4isc26::detail
