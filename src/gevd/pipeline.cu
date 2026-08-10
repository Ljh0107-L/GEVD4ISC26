#include "gevd4isc26/solver.hpp"

#include "diagnostics/timing.hpp"
#include "diagnostics/validation.hpp"
#include "distributed/context.hpp"
#include "evd/adapter.hpp"
#include "gevd/backtransform.hpp"
#include "gevd/cholesky.hpp"
#include "gevd/factor_cache.hpp"
#include "gevd/numerical_info.hpp"
#include "gevd/standard_reduction.hpp"
#include "gevd/workspace.hpp"
#include "runtime/cpu_affinity.hpp"
#include "runtime/resources.hpp"

#include <mpi.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gevd4isc26 {

DiagonalMatrix::DiagonalMatrix(std::vector<double> diagonal)
    : diagonal_(std::move(diagonal)) {}

std::int64_t DiagonalMatrix::order() const noexcept {
  return static_cast<std::int64_t>(diagonal_.size());
}

double DiagonalMatrix::operator()(std::int64_t row,
                                  std::int64_t column) const {
  if (row < 0 || row >= order() || column < 0 || column >= order()) {
    throw std::out_of_range("Lambda index is outside the matrix");
  }
  return row == column ? diagonal_[static_cast<std::size_t>(row)] : 0.0;
}

class GeneralizedEigensolver::Impl {
public:
  Impl(MPI_Comm communicator, ProcessGrid grid, SolverOptions options)
      : context_(communicator, grid, options.first_gpu), options_(options) {}

  [[nodiscard]] GeneralizedEigenResult solve(const DistributedMatrixView& A,
                                             const DistributedMatrixView& B) {
    detail::ScopedCpuAffinityRestore preserve_cpu_affinity;
    const double solve_start = MPI_Wtime();
    detail::validateGeneralizedEigenproblem(context_, A, B, options_);

    // ---------------------------------------------------------------------
    // Prepare distributed storage and one workspace shared by all stages.
    // ---------------------------------------------------------------------
    detail::GevdPipelineTimings timings;
    double stage_start = MPI_Wtime();
    const bool prepare_inverse_concurrently =
        options_.backtransform == BacktransformMethod::OverlappedInverse;

    const detail::FactorReuse reuse = factor_cache_.prepare(
        context_, B, A.leading_dimension, options_.cache_cholesky,
        options_.cache_inverse, prepare_inverse_concurrently);
    double* const cholesky_B = factor_cache_.cholesky();
    double* const inverse_transpose_B =
        prepare_inverse_concurrently ? factor_cache_.inverse() : nullptr;

    detail::GevdWorkspace workspace(
        context_, A, B, prepare_inverse_concurrently, eigenvector_cache_);
    detail::CholeskyStage cholesky(
        context_, workspace.order(), cholesky_B,
        workspace.solverDescriptorB());
    detail::StandardReductionStage reduction(
        context_, workspace.order(), workspace.standardMatrix().as<double>(),
        cholesky_B, workspace.solverDescriptorA(),
        workspace.solverDescriptorB());

    detail::WorkspaceSize workspace_size;
    if (!reuse.cholesky) {
      workspace_size.include(cholesky.queryWorkspace());
    }
    workspace_size.include(reduction.queryWorkspace());
    const detail::WorkspaceSize backtransform_workspace =
        prepare_inverse_concurrently
            ? detail::queryOverlapBacktransformWorkspace(
                  context_, workspace.order(), cholesky_B,
                  inverse_transpose_B, workspace.standardEigenvectors(),
                  workspace.generalizedEigenvectors(),
                  workspace.blasDescriptorB(), workspace.blasDescriptorQ(),
                  reuse.inverse)
            : detail::queryDirectBacktransformWorkspace(
                  context_, workspace.order(), cholesky_B,
                  workspace.standardEigenvectors(),
                  workspace.blasDescriptorB(), workspace.blasDescriptorQ());
    workspace_size.include(backtransform_workspace);
    workspace.allocateNumericalWorkspace(workspace_size);
    timings.resource_setup_seconds = MPI_Wtime() - stage_start;

    timings.reused_cholesky = reuse.cholesky;
    timings.reused_inverse = reuse.inverse;
    timings.used_overlapped_inverse = prepare_inverse_concurrently;

    // ---------------------------------------------------------------------
    // Stage 1: move A and, when not cached, B to this rank's GPU.
    // ---------------------------------------------------------------------
    stage_start = MPI_Wtime();
    workspace.uploadInputs(A, B, cholesky_B, reuse.cholesky);
    timings.input_transfer_seconds = MPI_Wtime() - stage_start;

    // ---------------------------------------------------------------------
    // Stage 2: B = L L^T.  A repeated solve may reuse the distributed L.
    // ---------------------------------------------------------------------
    if (!reuse.cholesky) {
      stage_start = MPI_Wtime();
      cholesky.factorize(workspace.deviceWorkspace(),
                         workspace.hostWorkspace(),
                         workspace.numericalInfo());
      const int numerical_info = detail::readNumericalInfo(
          context_, workspace.numericalInfo());
      detail::checkNumericalInfo(
          context_, numerical_info, "Cholesky factorization of B");
      factor_cache_.markCholeskyValid(options_.cache_cholesky);
      timings.cholesky_seconds = MPI_Wtime() - stage_start;
    }

    // ---------------------------------------------------------------------
    // Stage 3: C = L^{-1} A L^{-T}.  C is now a standard symmetric problem.
    // ---------------------------------------------------------------------
    stage_start = MPI_Wtime();
    reduction.reduce(workspace.deviceWorkspace(), workspace.hostWorkspace(),
                     workspace.numericalInfo());
    const int reduction_info = detail::readNumericalInfo(
        context_, workspace.numericalInfo());
    detail::checkNumericalInfo(
        context_, reduction_info, "generalized-to-standard reduction");
    timings.reduction_to_standard_seconds = MPI_Wtime() - stage_start;

    // ---------------------------------------------------------------------
    // Stage 4: solve C Y = Y Lambda.  Its internal SBR, BC, tridiagonal
    // eigensolve, SBR-back, BC-back, and GEMM timings are returned explicitly.
    // ---------------------------------------------------------------------
    std::unique_ptr<detail::OverlapInverseBacktransform> inverse_preparation;
    detail::ConcurrentWork concurrent_work;
    if (prepare_inverse_concurrently) {
      inverse_preparation =
          std::make_unique<detail::OverlapInverseBacktransform>(
              context_, workspace.distribution(),
              workspace.leadingDimension(), cholesky_B, inverse_transpose_B,
              workspace.blasDescriptorB(), workspace.blasDescriptorQ(),
              workspace.deviceWorkspace(), workspace.hostWorkspace(),
              reuse.inverse);
      concurrent_work.callback = inverse_preparation->callback();
      concurrent_work.user_data = inverse_preparation->callbackData();
    }

    detail::StandardEigensolverOptions standard_options;
    standard_options.block_size = options_.evd_block_size;
    standard_options.panel_size = options_.evd_panel_size;
    standard_options.enable_sbr_bc_pipeline =
        options_.stage_pipeline == StagePipelineMode::Automatic;
    standard_options.pipeline_sweeps = options_.stage_pipeline_sweeps;
    standard_options.collect_timings = options_.print_timing;
    // GPU collectives in the preceding stages may have selected the
    // GPU-local NUMA node for this calling thread. Restore the exact mask
    // supplied to this rank before entering the CPU-containing EVD stage.
    preserve_cpu_affinity.restoreNow();
    timings.standard_eigensolver = detail::solveStandardEigenproblem(
        context_, workspace.distribution(), workspace.leadingDimension(),
        workspace.standardMatrix(), workspace.standardEigenvectors(),
        workspace.eigenvalues(), standard_options,
        concurrent_work);

    if (prepare_inverse_concurrently) {
      inverse_preparation->wait();
      timings.inverse_preparation_seconds =
          inverse_preparation->inversePreparationSeconds();
      factor_cache_.markInverseValid(options_.cache_cholesky &&
                                     options_.cache_inverse);
    }

    // eigenvalues is already replicated by the standard eigensolver stage.

    // ---------------------------------------------------------------------
    // Stage 5: recover generalized eigenvectors Q = L^{-T} Y.
    // ---------------------------------------------------------------------
    double* result_on_device = workspace.standardEigenvectors();
    stage_start = MPI_Wtime();
    if (prepare_inverse_concurrently) {
      inverse_preparation->multiply(
          workspace.standardEigenvectors(),
          workspace.generalizedEigenvectors());
      result_on_device = workspace.generalizedEigenvectors();
    } else {
      detail::backtransformWithTriangularSolve(
          context_, workspace.order(), cholesky_B,
          workspace.standardEigenvectors(), workspace.blasDescriptorB(),
          workspace.blasDescriptorQ(), workspace.deviceWorkspace(),
          workspace.hostWorkspace());
    }
    context_.synchronize();
    timings.generalized_backtransform_seconds = MPI_Wtime() - stage_start;

    // ---------------------------------------------------------------------
    // Stage 6: return this rank's block-cyclic shard of Q to host memory.
    // Lambda remains replicated and is represented by its diagonal.
    // ---------------------------------------------------------------------
    stage_start = MPI_Wtime();
    workspace.downloadEigenvectors(result_on_device);
    timings.output_transfer_seconds = MPI_Wtime() - stage_start;
    timings.total_seconds = MPI_Wtime() - solve_start;

    if (options_.print_timing) {
      detail::printGevdPipelineTimings(context_, timings);
    }
    return GeneralizedEigenResult{
        workspace.releaseEigenvectors(),
        DiagonalMatrix(workspace.releaseEigenvalues())};
  }

  [[nodiscard]] int rank() const noexcept { return context_.rank(); }
  [[nodiscard]] int device() const noexcept { return context_.device(); }

private:
  detail::DistributedGpuContext context_;
  detail::FactorizationCache factor_cache_;
  detail::DeviceAllocation eigenvector_cache_;
  SolverOptions options_;
};

GeneralizedEigensolver::GeneralizedEigensolver(MPI_Comm communicator,
                                               ProcessGrid grid,
                                               SolverOptions options)
    : impl_(std::make_unique<Impl>(communicator, grid, options)) {}

GeneralizedEigensolver::~GeneralizedEigensolver() = default;
GeneralizedEigensolver::GeneralizedEigensolver(
    GeneralizedEigensolver&&) noexcept = default;
GeneralizedEigensolver& GeneralizedEigensolver::operator=(
    GeneralizedEigensolver&&) noexcept = default;

GeneralizedEigenResult GeneralizedEigensolver::solve(
    const DistributedMatrixView& A,
    const DistributedMatrixView& B) {
  return impl_->solve(A, B);
}

int GeneralizedEigensolver::rank() const noexcept {
  return impl_->rank();
}

int GeneralizedEigensolver::device() const noexcept {
  return impl_->device();
}

}  // namespace gevd4isc26
