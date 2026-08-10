#pragma once

#include "distributed/context.hpp"
#include "runtime/resources.hpp"
#include "gevd4isc26/distributed_matrix.hpp"

#include <cstdint>
#include <vector>

namespace gevd4isc26::detail {

struct StandardEigensolverOptions {
  int block_size = 32;
  int panel_size = 1024;
  bool enable_sbr_bc_pipeline = false;
  int pipeline_sweeps = 0;
  bool collect_timings = false;
};

// A callback can enqueue independent GEVD work while the standard EVD stage
// is finishing.  The optimized path uses it to prepare L^{-T}.
struct ConcurrentWork {
  int (*callback)(void*) = nullptr;
  void* user_data = nullptr;
};

struct StandardEigensolverTimings {
  double total_seconds = 0.0;
  double layout_storage_seconds = 0.0;
  double layout_to_column_block_seconds = 0.0;
  double engine_initialization_seconds = 0.0;
  double matrix_setup_seconds = 0.0;
  double dense_to_band_seconds = 0.0;
  double band_to_tridiagonal_seconds = 0.0;
  double reduction_critical_path_seconds = 0.0;
  double tridiagonal_eigensolve_seconds = 0.0;
  double dense_backtransform_seconds = 0.0;
  double band_backtransform_seconds = 0.0;
  double eigenvector_composition_seconds = 0.0;
  double overlapped_critical_path_seconds = 0.0;
  double numerical_wall_seconds = 0.0;
  double engine_cleanup_seconds = 0.0;
  double engine_call_seconds = 0.0;
  double concurrent_work_wait_seconds = 0.0;
  double layout_to_block_cyclic_seconds = 0.0;
};

// Input:  C = L^{-1} A L^{-T}, in the application's 2-D block-cyclic layout.
// Output: Y from C Y = Y Lambda, restored to that same layout.
//
// The optimized engine uses a block-aligned column layout internally.  C is
// released after the forward redistribution to keep peak GPU memory bounded.
StandardEigensolverTimings solveStandardEigenproblem(
    const DistributedGpuContext& context,
    const MatrixDistribution& distribution,
    std::int64_t block_cyclic_leading_dimension,
    DeviceAllocation& standard_matrix_C,
    double* eigenvectors_Y_block_cyclic,
    std::vector<double>& eigenvalues,
    const StandardEigensolverOptions& options,
    ConcurrentWork concurrent_work = {});

}  // namespace gevd4isc26::detail
