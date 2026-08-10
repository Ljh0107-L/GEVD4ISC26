#include "diagnostics/validation.hpp"

#include "runtime/error.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace gevd4isc26::detail {
namespace {

[[nodiscard]] bool isPowerOfTwo(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

void collectivelyValidate(const DistributedGpuContext& context,
                          const std::function<void()>& local_validation) {
  std::string local_error;
  try {
    local_validation();
  } catch (const std::exception& error) {
    local_error = error.what();
  }

  const int no_failure = context.size();
  int first_failing_rank = local_error.empty() ? no_failure : context.rank();
  int global_first_failing_rank = no_failure;
  GEVD_MPI(MPI_Allreduce(&first_failing_rank, &global_first_failing_rank, 1,
                         MPI_INT, MPI_MIN, context.communicator()));
  if (global_first_failing_rank == no_failure) {
    return;
  }

  int message_length = context.rank() == global_first_failing_rank
                           ? static_cast<int>(local_error.size())
                           : 0;
  GEVD_MPI(MPI_Bcast(&message_length, 1, MPI_INT, global_first_failing_rank,
                     context.communicator()));
  std::string message(static_cast<std::size_t>(message_length), '\0');
  if (context.rank() == global_first_failing_rank) {
    message = local_error;
  }
  GEVD_MPI(MPI_Bcast(message.data(), message_length, MPI_CHAR,
                     global_first_failing_rank, context.communicator()));
  throw std::invalid_argument("rank " +
                              std::to_string(global_first_failing_rank) +
                              " rejected the distributed problem: " + message);
}

void validateLayoutAcrossRanks(const DistributedGpuContext& context,
                               const MatrixDistribution& distribution) {
  const long long local_metadata[7] = {
      static_cast<long long>(distribution.order()),
      static_cast<long long>(distribution.rowBlockSize()),
      static_cast<long long>(distribution.columnBlockSize()),
      static_cast<long long>(distribution.grid().rows),
      static_cast<long long>(distribution.grid().columns),
      static_cast<long long>(distribution.grid().source_row),
      static_cast<long long>(distribution.grid().source_column)};
  long long minimum_metadata[7] = {};
  long long maximum_metadata[7] = {};
  GEVD_MPI(MPI_Allreduce(local_metadata, minimum_metadata, 7, MPI_LONG_LONG_INT,
                         MPI_MIN, context.communicator()));
  GEVD_MPI(MPI_Allreduce(local_metadata, maximum_metadata, 7, MPI_LONG_LONG_INT,
                         MPI_MAX, context.communicator()));
  for (int index = 0; index < 7; ++index) {
    if (minimum_metadata[index] != maximum_metadata[index]) {
      throw std::invalid_argument(
          "matrix order, block sizes, and process-grid metadata must match on every rank");
    }
  }
}

void validateOptionsAcrossRanks(const DistributedGpuContext& context,
                                const SolverOptions& options) {
  const int local_options[9] = {
      options.evd_block_size,
      options.evd_panel_size,
      options.first_gpu,
      static_cast<int>(options.backtransform),
      static_cast<int>(options.stage_pipeline),
      options.stage_pipeline_sweeps,
      options.cache_cholesky ? 1 : 0,
      options.cache_inverse ? 1 : 0,
      options.print_timing ? 1 : 0};
  int minimum_options[9] = {};
  int maximum_options[9] = {};
  GEVD_MPI(MPI_Allreduce(local_options, minimum_options, 9, MPI_INT, MPI_MIN,
                         context.communicator()));
  GEVD_MPI(MPI_Allreduce(local_options, maximum_options, 9, MPI_INT, MPI_MAX,
                         context.communicator()));
  for (int index = 0; index < 9; ++index) {
    if (minimum_options[index] != maximum_options[index]) {
      throw std::invalid_argument("all MPI ranks must use identical solver options");
    }
  }
}

}  // namespace

void validateGeneralizedEigenproblem(const DistributedGpuContext& context,
                                     const DistributedMatrixView& A,
                                     const DistributedMatrixView& B,
                                     const SolverOptions& options) {
  collectivelyValidate(context, [&] {
    A.validate();
    B.validate();
    if (!A.distribution.sameLayoutAs(B.distribution)) {
      throw std::invalid_argument("A and B must have identical block-cyclic layouts");
    }
    if (A.leading_dimension != B.leading_dimension) {
      throw std::invalid_argument(
          "A and B must use the same local leading dimension");
    }
    if (A.distribution.rank() != context.rank()) {
      throw std::invalid_argument("matrix distribution rank does not match MPI rank");
    }

    const ProcessGrid& matrix_grid = A.distribution.grid();
    const ProcessGrid& solver_grid = context.gridShape();
    if (matrix_grid.rows != solver_grid.rows ||
        matrix_grid.columns != solver_grid.columns ||
        matrix_grid.source_row != solver_grid.source_row ||
        matrix_grid.source_column != solver_grid.source_column) {
      throw std::invalid_argument("matrix and solver process grids do not match");
    }
    if (A.distribution.localRows() == 0 ||
        A.distribution.localColumns() == 0) {
      throw std::invalid_argument(
          "every rank must own at least one local row and column");
    }

    if (options.backtransform != BacktransformMethod::OverlappedInverse &&
        options.backtransform != BacktransformMethod::TriangularSolve) {
      throw std::invalid_argument(
          "unknown generalized-eigenvector backtransform method");
    }
    if (options.stage_pipeline != StagePipelineMode::Disabled &&
        options.stage_pipeline != StagePipelineMode::Automatic) {
      throw std::invalid_argument("unknown SBR-to-BC stage-pipeline mode");
    }
    if (options.stage_pipeline_sweeps < 0) {
      throw std::invalid_argument(
          "stage_pipeline_sweeps must be non-negative");
    }
    if (options.cache_inverse && !options.cache_cholesky) {
      throw std::invalid_argument("cache_inverse requires cache_cholesky");
    }

    const std::int64_t order = A.distribution.order();
    if (options.evd_block_size <= 0 || options.evd_panel_size <= 0 ||
        order % options.evd_block_size != 0 ||
        options.evd_panel_size % options.evd_block_size != 0 ||
        !isPowerOfTwo(options.evd_panel_size / options.evd_block_size)) {
      throw std::invalid_argument(
          "the distributed standard-eigensolver stage requires n % b == 0, "
          "nb % b == 0, and nb / b to be a power of two");
    }

    const int ranks = context.size();
    if (!isPowerOfTwo(ranks)) {
      throw std::invalid_argument(
          "the distributed standard-eigensolver stage requires a "
          "power-of-two MPI rank count (1, 2, 4, 8, 16, ...)");
    }
    if (order / options.evd_block_size < ranks) {
      throw std::invalid_argument(
          "matrix order must provide at least one EVD block per MPI rank");
    }

    const std::int64_t total_evd_blocks = order / options.evd_block_size;
    const std::int64_t minimum_evd_columns =
        (total_evd_blocks / ranks) * options.evd_block_size;
    const std::int64_t backtransform_block_columns =
        std::min<std::int64_t>(order, options.evd_panel_size);
    if (backtransform_block_columns > minimum_evd_columns) {
      throw std::invalid_argument(
          "min(n, nb) must fit in every rank's standard-EVD column block; "
          "reduce evd_panel_size or increase matrix order");
    }
    if (order > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(
          "matrix order exceeds the MPI count used for Lambda broadcast");
    }
  });

  validateLayoutAcrossRanks(context, A.distribution);
  validateOptionsAcrossRanks(context, options);
}

}  // namespace gevd4isc26::detail
