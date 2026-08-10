#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <mpi.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include <nccl.h>

#include "bc_backtransform.hpp"
#include "bulge_chasing_kernels.cuh"
#include "d_and_c_solver.hpp"
#include "panel_qr.hpp"
#include "runtime/cuda_check.hpp"
#include "utility_kernels.cuh"

#include "communication/layout_conversion.cuh"
#include "runtime/cpu_affinity.hpp"
#include "runtime/error_handling.hpp"
#include "engine/evd_types.hpp"
#include "interface/distributed_evd.h"
#include "stages/pipeline/stage_pipeline_plan.hpp"
#include "tuning/driver_config.hpp"

using namespace gevd4isc26::evd;

namespace {


#include "tuning/pipeline_guard.cuh"
#include "diagnostics/band_changes.cuh"
#include "diagnostics/tridiagonal_dump.cuh"
#include "runtime/cached_memory.cuh"
#include "communication/final_redistribution.cuh"

#include "diagnostics/matrix_io.cuh"
#include "stages/sbr/sbr_runtime.cuh"
#include "stages/sbr/single_block_sbr.cuh"
#include "stages/sbr/double_block_sbr.cuh"
#include "diagnostics/sbr_validation.cuh"
#include "communication/band_gather.cuh"
#include "diagnostics/band_validation.cuh"
#include "stages/bc/packed_u.cuh"
#include "communication/band_exchange.cuh"
#include "stages/bc/distributed_ranges.cuh"
#include "stages/bc/local_pipeline.cuh"
#include "stages/bc/segmented_stage.cuh"
#include "diagnostics/bc_validation.cuh"
#include "stages/pipeline/sbr_bc_pipeline.cuh"
#include "stages/backtransform/tridiagonal_extract.cuh"
#include "stages/backtransform/sbr_backtransform.cuh"
#include "stages/backtransform/eigenvector_composition.cuh"
#include "diagnostics/evd_verification.cuh"
#include "runtime/evd_runtime.cuh"

} // namespace

namespace {

int runDistributedStandardEigensolver(
    int64_t n64,
    double *d_a,
    int64_t ld_a64,
    int64_t local_cols64,
    double *eigvals,
    double *d_z,
    int64_t ld_z64,
    int64_t z_cols64,
    MPI_Comm comm,
    const gevd4isc26_standard_evd_options_t &options,
    gevd4isc26_concurrent_work_t concurrentWork,
    void *concurrentWorkData,
    gevd4isc26_standard_evd_timings_t *timings)
{
  gevd4isc26::detail::ScopedCpuAffinityRestore preserve_cpu_affinity;
  ErrorCommunicatorScope error_scope(comm);
  DistContext ctx;
  ctx.comm = comm;
  MPI_CHECK(MPI_Comm_rank(ctx.comm, &ctx.rank));
  MPI_CHECK(MPI_Comm_size(ctx.comm, &ctx.size));
  const double functionStart = MPI_Wtime();

  ctx.n = static_cast<long>(n64);
  ctx.b = static_cast<long>(options.block_size);
  ctx.nb = static_cast<long>(options.panel_size);
  const long ldA = static_cast<long>(ld_a64);
  const long ldZ = static_cast<long>(ld_z64);
  const long localCols = static_cast<long>(local_cols64);
  const long zCols = static_cast<long>(z_cols64);

  if (!isSupportedRankCount(ctx.size)) {
    if (ctx.rank == 0) {
      std::cerr << "The GEVD standard-eigensolver stage requires a "
                << "power-of-two number of MPI ranks (1, 2, 4, 8, 16, ...)"
                << std::endl;
    }
    return 2;
  }
  if (ctx.n <= 0 || ctx.b <= 0 || ctx.nb <= 0 || ctx.n % ctx.b != 0 ||
      ctx.nb % ctx.b != 0 || !isPowerOfTwo(ctx.nb / ctx.b)) {
    if (ctx.rank == 0) {
      std::cerr << "Invalid standard-EVD parameters: require n>0, b>0, nb>0, "
                << "n%b==0, nb%b==0, nb/b power-of-two" << std::endl;
    }
    return 2;
  }
  if (ldA != ctx.n || ldZ < ctx.n) {
    if (ctx.rank == 0) {
      std::cerr << "The standard-EVD device stage requires ld_a==n and ld_z>=n"
                << std::endl;
    }
    return 2;
  }

  buildBlockAlignedColumnDistribution(ctx.n, ctx.b, ctx.size, &ctx.counts, &ctx.displs);
  ctx.colStart = ctx.displs[ctx.rank];
  ctx.localCols = ctx.counts[ctx.rank];
  if (localCols != ctx.localCols || zCols != ctx.localCols) {
    if (ctx.rank == 0) {
      std::cerr << "The standard-EVD device stage requires local columns to match its "
                << "block-aligned column distribution" << std::endl;
    }
    return 2;
  }

  const StagePipelinePlan stagePipeline =
      buildStagePipelinePlan(ctx,
                             options.enable_sbr_bc_pipeline != 0,
                             options.pipeline_sweeps);
  const EvdDriverConfig config =
      EvdDriverConfig::load(stagePipeline.enabled, ctx.n, ctx.b);
  // Keep the eigenvector columns balanced by default.  Moving five percent
  // from one rank to another made BC Back's critical rank slower at n=32000
  // and erased the SBR/BC overlap.  The existing tuning hook remains for
  // machines whose asymmetric rank placement benefits from it.
  buildBackTransformDistribution(ctx.n,
                                 ctx.b,
                                 ctx.size,
                                 config.back_balance_percent,
                                 config.align_backtransform_to_block,
                                 &ctx.qCounts,
                                 &ctx.qDispls);
  ctx.qColStart = ctx.qDispls[ctx.rank];
  ctx.qLocalCols = ctx.qCounts[ctx.rank];
  for (int r = 0; r < ctx.size; ++r) {
    if (ctx.counts[r] % ctx.b != 0) {
      if (ctx.rank == 0) {
        std::cerr << "Each standard-EVD column block must be divisible by b"
                  << std::endl;
      }
      return 2;
    }
  }

  const long sbrPipelineGuardCols =
      config.stage_pipeline ? stagePipeline.sbrGuardCols : 0;

  EvdRuntime runtime(ctx);
  if (config.sbr_pipeline_early_return &&
      !config.sbr_suffix_communicator) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_STAGE_PIPELINE requires EVD_SBR_SUFFIX_COMM=1" << std::endl;
    }
    return 2;
  }
  if (config.initializeSuffixCommunicators()) {
    initSbrSuffixNccls(&ctx);
  }

  const cusolverDnHandle_t cusolver = runtime.cusolver();
  const cublasHandle_t cublas = runtime.cublas();

  if (ctx.rank == 0 && options.print_progress != 0) {
    std::cout << "GEVD4ISC26 standard symmetric eigensolver stage" << std::endl;
    std::cout << "order=" << ctx.n << ", block=" << ctx.b
              << ", panel=" << ctx.nb
              << ", ranks=" << ctx.size << std::endl;
    if (options.enable_sbr_bc_pipeline != 0 && !config.stage_pipeline) {
      std::cout << "SBR->BC stage-pipeline disabled: no non-empty overlap window"
                << std::endl;
    } else if (config.stage_pipeline) {
      std::cout << "SBR->BC stage-pipeline initial_sweeps="
                << stagePipeline.initialSweeps
                << " sbr_release_end=" << stagePipeline.sbrReleaseEnd
                << " bc_changed_band_end=" << stagePipeline.changedBandEnd
                << std::endl;
    }
  }

  double *dWLocal = nullptr;
  double *dYLocal = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dWLocal, sizeof(double) * static_cast<size_t>(ctx.n) * ctx.localCols));
  EVD_CUDA_CHECK(evdMalloc(&dYLocal, sizeof(double) * static_cast<size_t>(ctx.n) * ctx.localCols));
  EVD_CUDA_CHECK(cudaMemset(dWLocal, 0, sizeof(double) * static_cast<size_t>(ctx.n) * ctx.localCols));
  EVD_CUDA_CHECK(cudaMemset(dYLocal, 0, sizeof(double) * static_cast<size_t>(ctx.n) * ctx.localCols));

  MPI_CHECK(MPI_Barrier(ctx.comm));
  const double wallStart = MPI_Wtime();
  if (evd_redist::symmetrizeColumnBlockLowerToUpper(ctx.n,
                                                          d_a,
                                                          ctx.rank,
                                                          ctx.size,
                                                          ctx.counts,
                                                          ctx.displs,
                                                          ctx.nccl,
                                                          ctx.commStream) != 0) {
    MPI_Abort(ctx.comm, 1);
  }

  SbrWorkspace sbrWs;
  allocateSbrWorkspace(ctx, &sbrWs, config.sbr_double_block);

  double *dBand = nullptr;
  const long ldBand = 2 * ctx.b;
  if (config.sbr_band_pipeline) {
    EVD_CUDA_CHECK(evdMalloc(&dBand, sizeof(double) * static_cast<size_t>(ldBand) * ctx.n));
    EVD_CUDA_CHECK(cudaMemset(dBand, 0, sizeof(double) * static_cast<size_t>(ldBand) * ctx.n));
  }

  MPI_CHECK(MPI_Barrier(ctx.comm));
  const double sbrStart = MPI_Wtime();
  if (config.sbr_double_block) {
    distributedSbrDoubleBlock(ctx,
                              cusolver,
                              cublas,
                              d_a,
                              dWLocal,
                              dYLocal,
                              &sbrWs,
                              config.sbr_suffix_communicator,
                              dBand,
                              config.sbr_band_pipeline,
                              config.sbr_pipeline_early_return,
                              sbrPipelineGuardCols);
  } else {
    distributedSbr(ctx, cusolver, cublas, d_a, dWLocal, dYLocal, &sbrWs,
                   config.sbr_suffix_communicator);
  }
  EVD_CUDA_CHECK(cudaDeviceSynchronize());
  if (config.initializeSuffixCommunicators() && !config.stage_pipeline &&
      !config.bc_tile_suffix_broadcast) {
    destroySbrSuffixNccls(&ctx);
  }
  const double sbrEnd = MPI_Wtime();

  if (dBand == nullptr) {
    EVD_CUDA_CHECK(evdMalloc(&dBand, sizeof(double) * static_cast<size_t>(ldBand) * ctx.n));
  }

  if (!config.stage_pipeline) {
    MPI_CHECK(MPI_Barrier(ctx.comm));
  }
  const double bcStart = MPI_Wtime();
  if (!config.sbr_band_pipeline) {
    gatherBandToAll(ctx, d_a, dBand);
  }
  const long packedUElems = bcBacktransformPackedElementCount(ctx.n);
  double *dUPacked = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dUPacked, sizeof(double) * static_cast<size_t>(packedUElems)));
  PackedUOffsetTable packedUOffsets = createPackedUOffsetTable(ctx.n);
  if (config.stage_pipeline) {
    runBcStagePipelinePackedU(ctx,
                              stagePipeline,
                              dBand,
                              dUPacked,
                              packedUElems,
                              packedUOffsets);
  } else if (config.bc_local_pipeline) {
    runBcPipelineLocalRangesPackedU(ctx,
                                    dBand,
                                    dUPacked,
                                    packedUElems,
                                    packedUOffsets);
  } else if (config.bc_range_mode == 2) {
    runBcSegmentedPackedU(ctx,
                          dBand,
                          dUPacked,
                          packedUElems,
                          config.bc_brown_columns,
                          packedUOffsets);
  } else if (config.bc_range_mode != 0) {
    runBcDistributedRangesPackedU(ctx,
                                  dBand,
                                  dUPacked,
                                  packedUElems,
                                  config.bc_sync_mode,
                                  packedUOffsets);
  } else {
    runBcOnReplicatedBandPackedUFull(ctx,
                                     dBand,
                                     dUPacked,
                                     packedUElems,
                                     packedUOffsets);
  }
  EVD_CUDA_CHECK(cudaDeviceSynchronize());
  const double bcEnd = MPI_Wtime();

  int dcInfo = 0;
  d_and_c::AsyncTridiagonalSolve dcSolve;
  std::vector<double> hD;
  std::vector<double> hE;
  double dcTimeMs = 0.0;
  if (config.bc_range_mode == 2 && config.bc_extract_last_owner) {
    extractTridiagonalFromBandOwnerToRoot(ctx, ldBand, dBand, ctx.size - 1, &hD, &hE);
  } else if (config.bc_range_mode == 2 &&
             (config.bc_segmented_block_local ||
              config.bc_segmented_tile_wave ||
              config.bc_segmented_rolling)) {
    extractTridiagonalOwnedColumnsToRoot(ctx, ldBand, dBand, &hD, &hE);
  } else if (config.bc_range_mode == 2 &&
             !config.bc_segmented_full_band_sync &&
             !config.bc_segmented_brown_only) {
    extractTridiagonalFromBandOwnerToRoot(ctx, ldBand, dBand, ctx.size - 1, &hD, &hE);
  } else if (config.bc_local_pipeline || config.bc_neighbor_handoff ||
             config.bc_range_mode == 2 || config.bc_sync_mode == 2 ||
             config.bc_sync_mode == 3) {
    extractTridiagonalOwnedColumnsToRoot(ctx, ldBand, dBand, &hD, &hE);
  }
  if (ctx.rank == 0) {
    if (!config.bc_local_pipeline && !config.bc_neighbor_handoff &&
        config.bc_range_mode != 2 && config.bc_sync_mode != 2 &&
        config.bc_sync_mode != 3) {
      extractTridiagonalFromBand(ctx.n, ldBand, dBand, &hD, &hE);
    }
    dumpTridiagonalIfRequested(ctx.n, hD, hE);
    // SBR/BC communication may have narrowed the engine's calling thread to
    // the GPU-local NUMA node. Restore the rank's engine-entry mask before
    // allocating/first-touching the host eigenvector matrix and launching
    // MKL so the CPU D&C task can use every CPU granted to rank zero.
    preserve_cpu_affinity.restoreNow();
    dcInfo = dcSolve.start(ctx.n, std::move(hD), std::move(hE));
    if (dcInfo != 0) {
      std::cerr << "rank 0 failed to start CPU D&C, info="
                << dcInfo << std::endl;
    }
  }

  double *dQ = nullptr;
  // BC backtransform advances a fixed-size sliding window. Its final,
  // partially filled reflector tile still touches the padded tail, so the
  // leading dimension must cover one complete sweep plus one column tile.
  // Relying on nb alone was insufficient when nb=256 and made the last local
  // eigenvector column depend on allocator slack.
  const long bcBackPadding = std::max<long>(
      ctx.nb, kBcBackSweepRows + kBcBackColumnTile);
  const long ldQ = ctx.n + bcBackPadding;
  EVD_CUDA_CHECK(evdMalloc(&dQ, sizeof(double) * static_cast<size_t>(ldQ) * ctx.qLocalCols));
  dim3 idBlock(32, 8);
  dim3 idGrid(ceilDiv(ctx.n, idBlock.x), ceilDiv(ctx.qLocalCols, idBlock.y));
  setLocalIdentity<<<idGrid, idBlock>>>(dQ, ldQ, ctx.n, ctx.qColStart, ctx.qLocalCols);
  EVD_CUDA_CHECK(cudaGetLastError());

  if (config.backtransform_barriers) {
    MPI_CHECK(MPI_Barrier(ctx.comm));
  }
  const double sbrBackStart = MPI_Wtime();
  if (config.sbr_double_block_back) {
    distributedSbrBackDoubleBlock(ctx, cublas, dQ, ldQ, dWLocal, dYLocal, &sbrWs);
  } else {
    distributedSbrBack(ctx, cublas, dQ, ldQ, dWLocal, dYLocal, &sbrWs);
  }
  double *dQTransposed = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dQTransposed, sizeof(double) * static_cast<size_t>(ldQ) * ctx.qLocalCols));
  transposeDistributedColumnMatrix(ctx, dQ, ldQ, dQTransposed, ldQ);
  EVD_CUDA_CHECK(evdFree(dQ));
  dQ = dQTransposed;
  const double sbrBackEnd = MPI_Wtime();

  if (config.backtransform_barriers) {
    MPI_CHECK(MPI_Barrier(ctx.comm));
  }
  const double bcBackStart = MPI_Wtime();
  if (applyPackedBcBacktransform(dQ,
                                 ldQ,
                                 ctx.qLocalCols,
                                 dUPacked,
                                 packedUOffsets.device,
                                 static_cast<long>(
                                     packedUOffsets.host.size()),
                                 ctx.n,
                                 ctx.b) != 0) {
    MPI_Abort(ctx.comm, 1);
  }
  EVD_CUDA_CHECK(cudaDeviceSynchronize());
  destroyPackedUOffsetTable(&packedUOffsets);
  const double bcBackEnd = MPI_Wtime();

  if (concurrentWork != nullptr) {
    const int callbackInfo = concurrentWork(concurrentWorkData);
    int maxCallbackInfo = 0;
    MPI_CHECK(MPI_Allreduce(&callbackInfo, &maxCallbackInfo, 1, MPI_INT, MPI_MAX, ctx.comm));
    if (maxCallbackInfo != 0) {
      if (ctx.rank == 0) {
        std::cerr << "Concurrent GEVD work failed with code="
                  << maxCallbackInfo << std::endl;
      }
      return maxCallbackInfo;
    }
  }

  if (ctx.rank == 0) {
    if (dcInfo == 0) {
      dcInfo = dcSolve.wait();
      dcTimeMs = dcSolve.solveMilliseconds();
    }
    const std::vector<double>& eigenvalues = dcSolve.eigenvalues();
    if (dcInfo == 0 && static_cast<long>(eigenvalues.size()) == ctx.n) {
      std::memcpy(eigvals,
                  eigenvalues.data(),
                  sizeof(double) * static_cast<size_t>(ctx.n));
    } else if (dcInfo == 0) {
      dcInfo = d_and_c::kInvalidArgument;
    }
  }
  MPI_CHECK(MPI_Bcast(&dcInfo, 1, MPI_INT, 0, ctx.comm));
  if (dcInfo != 0) {
    if (ctx.rank == 0) {
      std::cerr << "CPU tridiagonal D&C failed with info="
                << dcInfo << std::endl;
    }
    return dcInfo;
  }
  MPI_CHECK(MPI_Bcast(eigvals, static_cast<int>(ctx.n), MPI_DOUBLE, 0, ctx.comm));

  if (config.final_gemm_tile_columns <= 0) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_FINAL_TILE_COLS must be positive" << std::endl;
    }
    return 2;
  }
  double *dQFinalRows = nullptr;
  const double finalGemmMs = runFinalGemmStreaming(ctx,
                                                  cublas,
                                                  dQ,
                                                  ldQ,
                                                  dcSolve.eigenvectors(),
                                                  config.final_gemm_tile_columns,
                                                  config.final_gemm_start_barrier,
                                                  true,
                                                  &dQFinalRows);
  redistributeFinalRowsToColumns(ctx, dQFinalRows, d_z, ldZ);
  const double wallEnd = MPI_Wtime();

  const double cleanupStart = wallEnd;
  if (dQFinalRows != nullptr) {
    EVD_CUDA_CHECK(evdFree(dQFinalRows));
  }
  EVD_CUDA_CHECK(evdFree(dQ));
  EVD_CUDA_CHECK(evdFree(dUPacked));
  EVD_CUDA_CHECK(evdFree(dBand));
  freeSbrWorkspace(&sbrWs);
  EVD_CUDA_CHECK(evdFree(dYLocal));
  EVD_CUDA_CHECK(evdFree(dWLocal));

  runtime.release();
  const double functionEnd = MPI_Wtime();

  if (timings != nullptr) {
    // One reduction returns all standard-stage measurements to rank zero.
    // These stages overlap, so individual durations and critical paths are
    // retained instead of presenting their sum as elapsed GEVD time.
    double localTimes[11] = {
        (wallStart - functionStart) * 1000.0,
        (sbrStart - wallStart) * 1000.0,
        (sbrEnd - sbrStart) * 1000.0,
        (bcEnd - bcStart) * 1000.0,
        (sbrBackEnd - sbrBackStart) * 1000.0,
        (bcBackEnd - bcBackStart) * 1000.0,
        finalGemmMs,
        (wallEnd - wallStart) * 1000.0,
        (bcEnd - sbrStart) * 1000.0,
        (functionEnd - cleanupStart) * 1000.0,
        (functionEnd - functionStart) * 1000.0,
    };
    double maximumTimes[11] = {};
    MPI_CHECK(MPI_Reduce(localTimes, maximumTimes, 11, MPI_DOUBLE, MPI_MAX, 0,
                         ctx.comm));
    const double localDcTimeMs = ctx.rank == 0 ? dcTimeMs : 0.0;
    double maximumDcTimeMs = 0.0;
    MPI_CHECK(MPI_Reduce(&localDcTimeMs, &maximumDcTimeMs, 1, MPI_DOUBLE,
                         MPI_MAX, 0, ctx.comm));
    if (ctx.rank == 0) {
      timings->initialization_ms = maximumTimes[0];
      timings->matrix_setup_ms = maximumTimes[1];
      timings->dense_to_band_ms = maximumTimes[2];
      timings->band_to_tridiagonal_ms = maximumTimes[3];
      timings->reduction_critical_path_ms = maximumTimes[8];
      timings->tridiagonal_eigensolve_ms = maximumDcTimeMs;
      timings->dense_backtransform_ms = maximumTimes[4];
      timings->band_backtransform_ms = maximumTimes[5];
      timings->eigenvector_composition_ms = maximumTimes[6];
      timings->overlapped_critical_path_ms =
          maximumTimes[2] + maximumTimes[3] +
          std::max(maximumDcTimeMs, maximumTimes[4] + maximumTimes[5]) +
          maximumTimes[6];
      timings->numerical_wall_ms = maximumTimes[7];
      timings->cleanup_ms = maximumTimes[9];
      timings->engine_call_ms = maximumTimes[10];
    }
  }
  return 0;
}

} // namespace

extern "C" int gevd4isc26_symmetric_evd_device(
    int64_t order,
    double *matrix_C,
    int64_t leading_dimension_C,
    int64_t local_columns_C,
    double *eigenvalues,
    double *eigenvectors_Y,
    int64_t leading_dimension_Y,
    int64_t local_columns_Y,
    MPI_Comm communicator,
    const gevd4isc26_standard_evd_options_t *options,
    gevd4isc26_concurrent_work_t concurrent_work,
    void *concurrent_work_data,
    gevd4isc26_standard_evd_timings_t *timings)
{
  if (options == nullptr) {
    return 2;
  }
  if (timings != nullptr) {
    *timings = {};
  }
  return runDistributedStandardEigensolver(
      order, matrix_C, leading_dimension_C, local_columns_C, eigenvalues,
      eigenvectors_Y, leading_dimension_Y, local_columns_Y, communicator,
      *options, concurrent_work, concurrent_work_data, timings);
}
