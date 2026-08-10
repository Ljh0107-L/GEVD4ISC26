#pragma once

#include "gevd4isc26/distributed_matrix.hpp"

#include <mpi.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace gevd4isc26 {

// Lambda is mathematically an n-by-n matrix, but only its diagonal is stored.
// This keeps the result scalable while retaining explicit matrix semantics.
class DiagonalMatrix {
public:
  explicit DiagonalMatrix(std::vector<double> diagonal);

  [[nodiscard]] std::int64_t order() const noexcept;
  [[nodiscard]] double operator()(std::int64_t row, std::int64_t column) const;
  [[nodiscard]] const std::vector<double>& diagonal() const noexcept { return diagonal_; }

private:
  std::vector<double> diagonal_;
};

struct GeneralizedEigenResult {
  DistributedMatrix Q;
  DiagonalMatrix Lambda;
};

enum class BacktransformMethod {
  // Prepare L^{-T} concurrently with the standard-EVD backtransform, then
  // compute Q=L^{-T}Y.  This is the optimized default.
  OverlappedInverse,

  // Lower-memory fallback: solve L^T Q=Y after the standard EVD stage.
  TriangularSolve,
};

enum class StagePipelineMode {
  Disabled,
  Automatic,
};

struct SolverOptions {
  // Block size for dense-to-band reduction and panel width for the
  // standard symmetric eigensolver stage.
  int evd_block_size = 32;
  int evd_panel_size = 1024;

  // Overlap the tail of SBR with a dependency-safe prefix of BC.  Automatic
  // chooses the initial BC sweep batch from n, b, nb, and the rank layout.
  StagePipelineMode stage_pipeline = StagePipelineMode::Disabled;

  // Optional expert override for the initial BC sweep batch.  Zero keeps the
  // automatic choice.  Values that eliminate the overlap window are reduced
  // to the largest safe batch.
  int stage_pipeline_sweeps = 0;

  // The GPU selected by a rank is (first_gpu + node_local_rank) % visible_gpus,
  // matching the DFTB+ path.
  int first_gpu = 0;

  BacktransformMethod backtransform = BacktransformMethod::OverlappedInverse;

  // Repeated calls with the same B reuse its GPU Cholesky factor and inverse.
  // Cache identity includes the distributed shape and a sampled bitwise
  // fingerprint, matching the original optimized path.
  bool cache_cholesky = true;
  bool cache_inverse = true;

  // Print one human-readable timing tree for the complete GEVD pipeline from
  // rank zero.  Values are maximum wall times across MPI ranks.
  bool print_timing = false;
};

// Solves A Q = B Q Lambda for real symmetric A and symmetric-positive-definite B.
//
// Each MPI rank owns one GPU and one BLACS-compatible block-cyclic shard of A
// and B. A and B are copied to the GPU and are never modified on the host.
// Q has the same distribution as A and B; Lambda is replicated on every rank.
class GeneralizedEigensolver {
public:
  GeneralizedEigensolver(MPI_Comm communicator, ProcessGrid grid, SolverOptions options = {});
  ~GeneralizedEigensolver();

  GeneralizedEigensolver(const GeneralizedEigensolver&) = delete;
  GeneralizedEigensolver& operator=(const GeneralizedEigensolver&) = delete;
  GeneralizedEigensolver(GeneralizedEigensolver&&) noexcept;
  GeneralizedEigensolver& operator=(GeneralizedEigensolver&&) noexcept;

  [[nodiscard]] GeneralizedEigenResult solve(const DistributedMatrixView& A,
                                             const DistributedMatrixView& B);

  [[nodiscard]] int rank() const noexcept;
  [[nodiscard]] int device() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gevd4isc26
