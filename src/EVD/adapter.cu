#include "evd/adapter.hpp"

#include "runtime/cpu_affinity.hpp"
#include "runtime/error.hpp"

#include "evd/communication/layout_conversion.cuh"
#include "evd/interface/distributed_evd.h"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gevd4isc26::detail {
namespace {

namespace standard_evd_layout = evd_redist;

[[nodiscard]] std::size_t matrixBytes(std::int64_t rows,
                                      std::int64_t columns) {
  if (rows < 0 || columns < 0) {
    throw std::invalid_argument("negative matrix dimension");
  }
  const auto unsigned_rows = static_cast<std::size_t>(rows);
  const auto unsigned_columns = static_cast<std::size_t>(columns);
  if (unsigned_columns != 0 &&
      unsigned_rows > static_cast<std::size_t>(-1) / unsigned_columns) {
    throw std::overflow_error("column-block matrix size overflows size_t");
  }
  const auto elements = unsigned_rows * unsigned_columns;
  if (elements > static_cast<std::size_t>(-1) / sizeof(double)) {
    throw std::overflow_error("column-block matrix byte size overflows size_t");
  }
  return elements * sizeof(double);
}

[[nodiscard]] double seconds(double milliseconds) {
  return milliseconds / 1000.0;
}

}  // namespace

StandardEigensolverTimings solveStandardEigenproblem(
    const DistributedGpuContext& context,
    const MatrixDistribution& distribution,
    std::int64_t block_cyclic_leading_dimension,
    DeviceAllocation& standard_matrix_C,
    double* eigenvectors_Y_block_cyclic,
    std::vector<double>& eigenvalues,
    const StandardEigensolverOptions& options,
    ConcurrentWork concurrent_work) {
  ScopedCpuAffinityRestore preserve_cpu_affinity;
  const auto order = distribution.order();
  const int rank = context.rank();
  const int ranks = context.size();
  StandardEigensolverTimings result;
  const double total_start = MPI_Wtime();

  std::vector<long> column_counts;
  std::vector<long> column_offsets;
  standard_evd_layout::buildBlockAlignedColumnDistribution(
      static_cast<long>(order), static_cast<long>(options.block_size), ranks,
      &column_counts, &column_offsets);
  const auto local_columns =
      static_cast<std::int64_t>(column_counts.at(rank));
  double start = MPI_Wtime();
  DeviceAllocation column_block(matrixBytes(order, local_columns));
  result.layout_storage_seconds += MPI_Wtime() - start;

  start = MPI_Wtime();
  const int forward_status = standard_evd_layout::blacsToColumnBlock(
      static_cast<long>(order),
      static_cast<long>(distribution.rowBlockSize()),
      static_cast<long>(distribution.columnBlockSize()),
      distribution.grid().source_row, distribution.grid().source_column,
      distribution.grid().rows, distribution.grid().columns, rank, ranks,
      standard_matrix_C.as<const double>(),
      static_cast<long>(block_cyclic_leading_dimension),
      column_block.as<double>(), static_cast<long>(options.block_size),
      context.nccl(), context.stream());
  if (forward_status != 0) {
    throw std::runtime_error(
        "block-cyclic to standard-EVD column-block redistribution failed");
  }
  context.synchronize();
  result.layout_to_column_block_seconds = MPI_Wtime() - start;

  // C is no longer needed in block-cyclic form.  Releasing it here is a
  // deliberate large-problem memory optimization.
  start = MPI_Wtime();
  standard_matrix_C.reset();
  result.layout_storage_seconds += MPI_Wtime() - start;

  gevd4isc26_standard_evd_options_t engine_options{};
  engine_options.block_size = options.block_size;
  engine_options.panel_size = options.panel_size;
  engine_options.enable_sbr_bc_pipeline =
      options.enable_sbr_bc_pipeline ? 1 : 0;
  engine_options.pipeline_sweeps = options.pipeline_sweeps;
  engine_options.print_progress = options.collect_timings ? 1 : 0;

  gevd4isc26_standard_evd_timings_t engine_timings{};
  start = MPI_Wtime();
  // The forward NCCL redistribution can narrow the caller after the GEVD
  // layer restored it. Reapply this function's entry mask immediately before
  // the standard-EVD engine snapshots its own D&C affinity.
  preserve_cpu_affinity.restoreNow();
  const int evd_status = gevd4isc26_symmetric_evd_device(
      order, column_block.as<double>(), order, local_columns,
      eigenvalues.data(), column_block.as<double>(), order, local_columns,
      context.communicator(), &engine_options, concurrent_work.callback,
      concurrent_work.user_data,
      options.collect_timings ? &engine_timings : nullptr);
  if (evd_status != 0) {
    throw std::runtime_error(
        "distributed standard eigensolver failed with status " +
        std::to_string(evd_status));
  }
  result.engine_call_seconds = MPI_Wtime() - start;

  // The concurrent callback uses the outer GEVD stream; the standard EVD
  // engine owns separate streams.  Join them before the reverse layout change.
  start = MPI_Wtime();
  context.synchronize();
  result.concurrent_work_wait_seconds = MPI_Wtime() - start;

  start = MPI_Wtime();
  const int reverse_status = standard_evd_layout::columnBlockToBlacs(
      static_cast<long>(order),
      static_cast<long>(distribution.rowBlockSize()),
      static_cast<long>(distribution.columnBlockSize()),
      distribution.grid().source_row, distribution.grid().source_column,
      distribution.grid().rows, distribution.grid().columns, rank, ranks,
      column_block.as<const double>(), static_cast<long>(options.block_size),
      eigenvectors_Y_block_cyclic,
      static_cast<long>(block_cyclic_leading_dimension), context.nccl(),
      context.stream());
  if (reverse_status != 0) {
    throw std::runtime_error(
        "standard-EVD column-block to block-cyclic redistribution failed");
  }
  context.synchronize();
  result.layout_to_block_cyclic_seconds = MPI_Wtime() - start;
  start = MPI_Wtime();
  column_block.reset();
  result.layout_storage_seconds += MPI_Wtime() - start;
  result.total_seconds = MPI_Wtime() - total_start;

  if (options.collect_timings && rank == 0) {
    result.engine_initialization_seconds =
        seconds(engine_timings.initialization_ms);
    result.matrix_setup_seconds = seconds(engine_timings.matrix_setup_ms);
    result.dense_to_band_seconds = seconds(engine_timings.dense_to_band_ms);
    result.band_to_tridiagonal_seconds =
        seconds(engine_timings.band_to_tridiagonal_ms);
    result.reduction_critical_path_seconds =
        seconds(engine_timings.reduction_critical_path_ms);
    result.tridiagonal_eigensolve_seconds =
        seconds(engine_timings.tridiagonal_eigensolve_ms);
    result.dense_backtransform_seconds =
        seconds(engine_timings.dense_backtransform_ms);
    result.band_backtransform_seconds =
        seconds(engine_timings.band_backtransform_ms);
    result.eigenvector_composition_seconds =
        seconds(engine_timings.eigenvector_composition_ms);
    result.overlapped_critical_path_seconds =
        seconds(engine_timings.overlapped_critical_path_ms);
    result.numerical_wall_seconds = seconds(engine_timings.numerical_wall_ms);
    result.engine_cleanup_seconds = seconds(engine_timings.cleanup_ms);
    // Prefer the engine's synchronized, max-rank call measurement in reports.
    result.engine_call_seconds = seconds(engine_timings.engine_call_ms);
  }
  return result;
}

}  // namespace gevd4isc26::detail
