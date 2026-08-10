#include "diagnostics/timing.hpp"

#include "runtime/error.hpp"

#include <mpi.h>

#include <array>
#include <cstdio>

namespace gevd4isc26::detail {
namespace {

enum TimingIndex : std::size_t {
  Total,
  ResourceSetup,
  InputTransfer,
  Cholesky,
  ReductionToStandard,
  StandardTotal,
  StandardLayoutStorage,
  LayoutToColumnBlock,
  EngineInitialization,
  StandardMatrixSetup,
  DenseToBand,
  BandToTridiagonal,
  ReductionCriticalPath,
  TridiagonalEigensolve,
  DenseBacktransform,
  BandBacktransform,
  EigenvectorComposition,
  StandardOverlappedCriticalPath,
  StandardNumericalWall,
  EngineCleanup,
  StandardEngineCall,
  ConcurrentWorkWait,
  LayoutToBlockCyclic,
  InversePreparation,
  GeneralizedBacktransform,
  OutputTransfer,
  Count
};

using TimingArray = std::array<double, TimingIndex::Count>;

TimingArray flatten(const GevdPipelineTimings& timing) {
  const auto& standard = timing.standard_eigensolver;
  return {
      timing.total_seconds,
      timing.resource_setup_seconds,
      timing.input_transfer_seconds,
      timing.cholesky_seconds,
      timing.reduction_to_standard_seconds,
      standard.total_seconds,
      standard.layout_storage_seconds,
      standard.layout_to_column_block_seconds,
      standard.engine_initialization_seconds,
      standard.matrix_setup_seconds,
      standard.dense_to_band_seconds,
      standard.band_to_tridiagonal_seconds,
      standard.reduction_critical_path_seconds,
      standard.tridiagonal_eigensolve_seconds,
      standard.dense_backtransform_seconds,
      standard.band_backtransform_seconds,
      standard.eigenvector_composition_seconds,
      standard.overlapped_critical_path_seconds,
      standard.numerical_wall_seconds,
      standard.engine_cleanup_seconds,
      standard.engine_call_seconds,
      standard.concurrent_work_wait_seconds,
      standard.layout_to_block_cyclic_seconds,
      timing.inverse_preparation_seconds,
      timing.generalized_backtransform_seconds,
      timing.output_transfer_seconds,
  };
}

void line(const char* label, double seconds, const char* suffix = "") {
  std::printf("  %-56s %10.6f s%s\n", label, seconds, suffix);
}

}  // namespace

void printGevdPipelineTimings(const DistributedGpuContext& context,
                              const GevdPipelineTimings& local) {
  const TimingArray local_values = flatten(local);
  TimingArray maximum_values{};
  GEVD_MPI(MPI_Reduce(local_values.data(), maximum_values.data(),
                      static_cast<int>(maximum_values.size()), MPI_DOUBLE,
                      MPI_MAX, 0, context.communicator()));
  if (context.rank() != 0) {
    return;
  }

  std::printf("GEVD4ISC26 complete-pipeline timing "
              "(maximum wall time across %d MPI ranks)\n",
              context.size());
  line("total: solve A Q = B Q Lambda", maximum_values[Total]);
  line("1. prepare distributed descriptors and workspaces",
       maximum_values[ResourceSetup]);
  line("2. copy A and B to GPUs", maximum_values[InputTransfer],
       local.reused_cholesky ? "  [B factor reused]" : "");
  line("3. Cholesky factorization: B = L L^T",
       maximum_values[Cholesky],
       local.reused_cholesky ? "  [cached]" : "");
  line("4. generalized-to-standard reduction: C = L^-1 A L^-T",
       maximum_values[ReductionToStandard]);
  line("5. standard symmetric eigensolve: C Y = Y Lambda",
       maximum_values[StandardTotal]);
  line("   layout workspace allocation/release",
       maximum_values[StandardLayoutStorage]);
  line("   5.1 block-cyclic -> block-aligned columns",
       maximum_values[LayoutToColumnBlock]);
  line("   5.2 initialize distributed standard-EVD resources",
       maximum_values[EngineInitialization]);
  line("   5.3 complete symmetry and prepare reduction workspace",
       maximum_values[StandardMatrixSetup]);
  line("   5.4 dense -> band reduction (SBR)",
       maximum_values[DenseToBand]);
  line("   5.5 band -> tridiagonal reduction (BC)",
       maximum_values[BandToTridiagonal]);
  line("       SBR+BC critical path",
       maximum_values[ReductionCriticalPath]);
  line("   5.6 tridiagonal eigensolve (divide and conquer)",
       maximum_values[TridiagonalEigensolve]);
  line("   5.7 apply dense-to-band reflectors (SBR back)",
       maximum_values[DenseBacktransform]);
  line("   5.8 apply band-to-tridiagonal reflectors (BC back)",
       maximum_values[BandBacktransform]);
  line("   5.9 compose standard eigenvectors Y (GEMM)",
       maximum_values[EigenvectorComposition]);
  line("       standard-stage overlapped critical path",
       maximum_values[StandardOverlappedCriticalPath]);
  line("       numerical-core wall time",
       maximum_values[StandardNumericalWall]);
  line("   5.10 release standard-EVD resources",
       maximum_values[EngineCleanup]);
  line("       complete standard-EVD engine call",
       maximum_values[StandardEngineCall]);
  line("   5.11 block-aligned columns -> block-cyclic",
       maximum_values[LayoutToBlockCyclic]);
  if (local.used_overlapped_inverse) {
    line("6. prepare L^-T concurrently with stage 5",
         maximum_values[InversePreparation],
         local.reused_inverse ? "  [cached]" : "  [overlapped]");
    line("7. generalized eigenvectors: Q = L^-T Y (GEMM)",
         maximum_values[GeneralizedBacktransform]);
  } else {
    line("6. generalized eigenvectors: solve L^T Q = Y (TRSM)",
         maximum_values[GeneralizedBacktransform]);
  }
  line("8. copy distributed Q to host", maximum_values[OutputTransfer]);
  if (maximum_values[ConcurrentWorkWait] > 0.0) {
    line("   residual wait for concurrent L^-T preparation",
         maximum_values[ConcurrentWorkWait]);
  }
  std::fflush(stdout);
}

}  // namespace gevd4isc26::detail
