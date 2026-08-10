void runBcSegmentedPackedU(const DistContext &ctx,
                           double *dBand,
                           double *dUPacked,
                           long packedUElems,
                           long brownCols,
                           const PackedUOffsetTable &packedUOffsets)
{
  if (ctx.b != kBcBandwidth) {
    std::cerr << "EVD currently instantiates the existing BC kernel for b=32; got b="
              << ctx.b << std::endl;
    MPI_Abort(ctx.comm, 1);
  }
  const bool stagePipelineRequested =
      envIntOrDefault("EVD_STAGE_PIPELINE", 0) != 0;
  const bool segmentedFullBandSync = envIntOrDefault("EVD_BC_SEGMENTED_FULL_BCAST", 1) != 0;
  const bool segmentedBrownOnly = envIntOrDefault("EVD_BC_SEGMENTED_BROWN_ONLY", 0) != 0;
  const bool segmentedPackedUChain =
      envIntOrDefault("EVD_BC_SEGMENTED_PACKEDU_CHAIN", 1) != 0;
  const bool segmentedPackedUCompactChain =
      envIntOrDefault("EVD_BC_SEGMENTED_PACKEDU_COMPACT_CHAIN", 1) != 0;
  const bool segmentedPackedUOwnerBcast =
      envIntOrDefault("EVD_BC_SEGMENTED_PACKEDU_OWNER_BCAST", 0) != 0;
  const bool segmentedFlatStop = envIntOrDefault("EVD_BC_SEGMENTED_FLAT_STOP", 0) != 0;
  const bool segmentedComChain = envIntOrDefault("EVD_BC_SEGMENTED_COM_CHAIN", 1) != 0;
  const bool segmentedPullGrouped =
      envIntOrDefault("EVD_BC_SEGMENTED_PULL_GROUPED", 0) != 0;
  const bool printSegmentedBatch = envIntOrDefault("EVD_BC_SEGMENTED_PRINT_BATCH", 0) != 0;
  const bool debugStageProgress = debugStageProgressEnabled();
  const bool segmentedRolling = envIntOrDefault("EVD_BC_SEGMENTED_ROLLING", 0) != 0;
  const bool segmentedRollingBackSync =
      envIntOrDefault("EVD_BC_SEGMENTED_ROLLING_BACK_SYNC", 1) != 0;
  const bool segmentedRollingFlatStop =
      envIntOrDefault("EVD_BC_SEGMENTED_ROLLING_FLAT_STOP", 0) != 0;
  const bool segmentedRollingFullBcast =
      envIntOrDefault("EVD_BC_SEGMENTED_ROLLING_FULL_BCAST", 0) != 0;
  const bool segmentedRollingFullComBcast =
      envIntOrDefault("EVD_BC_SEGMENTED_ROLLING_FULL_COM_BCAST", 0) != 0;
  const bool segmentedRollingDirtyHandoff =
      envIntOrDefault("EVD_BC_SEGMENTED_ROLLING_DIRTY_HANDOFF", 0) != 0;
  const bool segmentedRollingDirtyBcast =
      envIntOrDefault("EVD_BC_SEGMENTED_ROLLING_DIRTY_BCAST", 0) != 0;
  const bool segmentedRollingBlockStop =
      envIntOrDefault("EVD_BC_SEGMENTED_ROLLING_BLOCK_STOP", 0) != 0;
  const bool segmentedRollingFrontierWindow =
      envIntOrDefault("EVD_BC_SEGMENTED_ROLLING_FRONTIER_WINDOW", 0) != 0;
  const bool segmentedRollingDirtyPrefix =
      envIntOrDefault("EVD_BC_SEGMENTED_ROLLING_DIRTY_PREFIX", 0) != 0;
  const bool segmentedRollingPullHandoff =
      envIntOrDefault("EVD_BC_SEGMENTED_ROLLING_PULL_HANDOFF", 0) != 0;
  const bool segmentedRollingDirtyChainAll =
      envIntOrDefault("EVD_BC_SEGMENTED_ROLLING_DIRTY_CHAIN_ALL", 1) != 0;
  const int defaultRollingDirtyChainMode =
      stagePipelineRequested ? 4 : (segmentedRollingDirtyChainAll ? 1 : 0);
  int segmentedRollingDirtyChainMode =
      envIntOrDefault("EVD_BC_SEGMENTED_ROLLING_DIRTY_CHAIN_MODE",
                      defaultRollingDirtyChainMode);
  if (segmentedRollingDirtyChainMode < 0 || segmentedRollingDirtyChainMode > 5) {
    segmentedRollingDirtyChainMode = defaultRollingDirtyChainMode;
  }
  const bool segmentedRowPipeline =
      envIntOrDefault("EVD_BC_SEGMENTED_ROW_PIPELINE", 0) != 0;
  const bool segmentedRowPipelineFullBcast =
      envIntOrDefault("EVD_BC_SEGMENTED_ROW_PIPELINE_FULL_BCAST", 1) != 0;
  const bool segmentedTileWave =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_WAVE", 0) != 0;
  const bool segmentedTileWaveFullBcast =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_FULL_BCAST", 1) != 0;
  const bool segmentedTileBatchMajor =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_BATCH_MAJOR", 0) != 0;
  const int segmentedTileStopMode =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_STOP_MODE", 0);
  const bool segmentedTileSingleStep =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_SINGLE_STEP", 0) != 0;
  const bool segmentedTileMultiStep =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_MULTI_STEP", 0) != 0;
  const bool segmentedTileSweepOwner =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_SWEEP_OWNER", 0) != 0;
  const bool segmentedTileRangeBcast =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_RANGE_BCAST", 0) != 0;
  const bool segmentedTileRangeHandoff =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_RANGE_HANDOFF", 0) != 0;
  const bool segmentedTileSuffixBcast =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_SUFFIX_BCAST", 0) != 0;
  const bool segmentedTileOwnerHandoff =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_OWNER_HANDOFF", 0) != 0;
  const bool segmentedTileOwnerHandoffPrefix =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_OWNER_HANDOFF_PREFIX", 0) != 0;
  const bool segmentedTileSparseComSync =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_SPARSE_COM_SYNC", 1) != 0;
  const bool segmentedTileRowOwnerPackedUBcast =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_ROW_OWNER_PACKEDU_BCAST", 0) != 0;
  const bool segmentedTileProgressPackedUBcast =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_PROGRESS_PACKEDU_BCAST", 0) != 0;
  const bool segmentedTileProgressPackedUImmediate =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_PROGRESS_PACKEDU_IMMEDIATE", 0) != 0;
  const bool segmentedTileFrontierRange =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_FRONTIER_RANGE", 0) != 0;
  const bool segmentedTileWrapGather =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_WRAP_GATHER", 1) != 0;
  const bool segmentedTilePullHandoff =
      envIntOrDefault("EVD_BC_SEGMENTED_TILE_PULL_HANDOFF", 0) != 0;
  const bool segmentedOwnerGatherBack =
      envIntOrDefault("EVD_BC_SEGMENTED_OWNER_GATHER_BACK", 1) != 0;
  const bool segmentedBlockLocal =
      envIntOrDefault("EVD_BC_SEGMENTED_BLOCK_LOCAL", 0) != 0;
  const bool segmentedBlockLocalFullBcast =
      envIntOrDefault("EVD_BC_SEGMENTED_BLOCK_LOCAL_FULL_BCAST", 0) != 0;
  const bool segmentedBlockLocalSlanted =
      envIntOrDefault("EVD_BC_SEGMENTED_BLOCK_LOCAL_SLANTED", 0) != 0;
  const bool segmentedBlockLocalSingleStep =
      envIntOrDefault("EVD_BC_SEGMENTED_BLOCK_LOCAL_SINGLE_STEP", 1) != 0;
  const bool segmentedOwnerSerialPackedUOwnerBcast =
      segmentedPackedUOwnerBcast && segmentedPackedUChain &&
      segmentedPackedUCompactChain && !segmentedBrownOnly && !segmentedFlatStop &&
      !segmentedRolling && !segmentedRowPipeline && !segmentedTileWave &&
      !segmentedBlockLocal;
  const long debugChangedColumnsMaxN =
      static_cast<long>(envIntOrDefault("EVD_DEBUG_BC_SEGMENTED_CHANGED_COLUMNS_MAX_N", 4096));
  const bool debugChangedColumns =
      envIntOrDefault("EVD_DEBUG_BC_SEGMENTED_CHANGED_COLUMNS", 0) != 0 &&
      ctx.n <= debugChangedColumnsMaxN;
  const double debugChangedColumnsEps =
      envDoubleOrDefault("EVD_DEBUG_BC_SEGMENTED_CHANGED_COLUMNS_EPS", 1e-12);
  long debugChangedColumnsMaxPrints =
      static_cast<long>(envIntOrDefault("EVD_DEBUG_BC_SEGMENTED_CHANGED_COLUMNS_MAX_PRINTS", 16));
  if (debugChangedColumnsMaxPrints < 0) {
    debugChangedColumnsMaxPrints = 0;
  }
  const bool flatStopUsesBrown =
      envIntOrDefault("EVD_BC_SEGMENTED_FLAT_BROWN", 1) != 0;
  long requestedBatchSweeps =
      static_cast<long>(envIntOrDefault("EVD_BC_SEGMENTED_BATCH_SWEEPS", 0));
  if (requestedBatchSweeps < 0) {
    requestedBatchSweeps = 0;
  }
  const long brownCapacitySweeps = std::max<long>(1, brownCols / (2 * ctx.b) + 1);
  const long effectiveBatchSweeps =
      (requestedBatchSweeps > 0) ? requestedBatchSweeps : brownCapacitySweeps;
  const long safeRollingDirtyBatchLimit =
      (brownCols >= 4 * ctx.b) ? 4 : brownCapacitySweeps;
  const long safeRollingBatchLimit =
	      segmentedRollingBlockStop ? brownCapacitySweeps
	                                : safeRollingDirtyBatchLimit;
	  const bool safeRollingDirtyChainMode =
	      !segmentedRollingDirtyHandoff ||
	      segmentedRollingDirtyChainMode == 1 ||
	      segmentedRollingDirtyChainMode == 4;
	  const bool safeSegmentedRollingMode =
	      segmentedRolling && segmentedRollingBackSync &&
	      !segmentedRollingFlatStop && !segmentedRollingFullBcast &&
	      !segmentedRollingFullComBcast && segmentedRollingDirtyHandoff &&
	      !segmentedRollingDirtyBcast && segmentedComChain &&
	      safeRollingDirtyChainMode &&
	      effectiveBatchSweeps >= 1 &&
	      effectiveBatchSweeps <= safeRollingBatchLimit;
  const bool safeSegmentedTileSweepOwnerMode =
      segmentedTileWave && (segmentedTileSingleStep || segmentedTileMultiStep) &&
      segmentedTileSweepOwner && !segmentedTileWaveFullBcast &&
      ((segmentedTileRangeBcast && !segmentedTileOwnerHandoff) ||
       (segmentedTileOwnerHandoff && segmentedTileBatchMajor && !segmentedTileRangeBcast));
  const bool safeSegmentedTilePullMode =
      segmentedTileWave && (segmentedTileSingleStep || segmentedTileMultiStep) &&
      !segmentedTileSweepOwner && !segmentedTileWaveFullBcast &&
      segmentedTileBatchMajor && segmentedTileRangeBcast &&
      segmentedTileRangeHandoff && segmentedTilePullHandoff;
  const bool safeSegmentedTileMode =
      safeSegmentedTileSweepOwnerMode || safeSegmentedTilePullMode;
  const bool safeSegmentedBlockLocalMode =
      segmentedBlockLocal && !segmentedBlockLocalFullBcast && brownCols >= ctx.n;
  const bool unsafeSegmentedMode =
      segmentedBrownOnly || segmentedFlatStop ||
      segmentedRollingFrontierWindow ||
      (segmentedRolling && segmentedRollingPullHandoff) ||
      (segmentedRolling && !safeSegmentedRollingMode) ||
      segmentedRowPipeline || (segmentedTileWave && !safeSegmentedTileMode) ||
      (segmentedBlockLocal && !safeSegmentedBlockLocalMode);
  if (unsafeSegmentedMode && envIntOrDefault("EVD_ALLOW_UNSAFE_BC_SEGMENTED", 0) == 0) {
    if (ctx.rank == 0) {
      if (segmentedRolling && segmentedRollingDirtyHandoff &&
          !safeRollingDirtyChainMode) {
        std::cerr << "rolling dirty brown handoff safe mode currently allows "
	                  << "EVD_BC_SEGMENTED_ROLLING_DIRTY_CHAIN_MODE=1 or 4 only; got "
	                  << segmentedRollingDirtyChainMode << std::endl;
	      }
      std::cerr << "EVD_BC_RANGE=2 safe wavefront supports full-band or neighbor full-band "
                << "handoff, rolling dirty brown handoff, tile sweep-owner "
                << "range-bcast/owner-handoff, and "
                << "tile row-owner pull-handoff with kernel sub-batching, plus block-local suffix mode "
                << "with EVD_BC_BROWN_COLS >= n. "
                << "brown-only/flat-stop/frontier-window/rolling/other tile diagnostics are unsafe; set "
                << "EVD_ALLOW_UNSAFE_BC_SEGMENTED=1 only for those diagnostics"
                << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  if (brownCols <= 0) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_BC_BROWN_COLS must be positive; got " << brownCols << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  if (segmentedTileWave && segmentedRolling) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_BC_SEGMENTED_TILE_WAVE and EVD_BC_SEGMENTED_ROLLING are "
                << "mutually exclusive diagnostics" << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  const int segmentedWaveModeCount = (segmentedRolling ? 1 : 0) +
                                     (segmentedRowPipeline ? 1 : 0) +
                                     (segmentedTileWave ? 1 : 0) +
                                     (segmentedBlockLocal ? 1 : 0);
  if (segmentedWaveModeCount > 1) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_BC_SEGMENTED_ROLLING, EVD_BC_SEGMENTED_ROW_PIPELINE, "
                << "EVD_BC_SEGMENTED_TILE_WAVE, and EVD_BC_SEGMENTED_BLOCK_LOCAL "
                << "are mutually exclusive diagnostics"
                << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  if (segmentedRowPipeline && !segmentedRowPipelineFullBcast) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_BC_SEGMENTED_ROW_PIPELINE currently requires "
                << "EVD_BC_SEGMENTED_ROW_PIPELINE_FULL_BCAST=1 until the brown-block "
                << "handoff window is verified" << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  if (segmentedTileWave && !segmentedTileWaveFullBcast &&
      !segmentedTileSingleStep && !segmentedTileMultiStep) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_BC_SEGMENTED_TILE_WAVE currently requires "
                << "EVD_BC_SEGMENTED_TILE_FULL_BCAST=1 unless "
                << "EVD_BC_SEGMENTED_TILE_SINGLE_STEP=1 or "
                << "EVD_BC_SEGMENTED_TILE_MULTI_STEP=1 is used for the verified "
                << "range handoff diagnostic" << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  if (segmentedTileSingleStep && segmentedTileMultiStep) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_BC_SEGMENTED_TILE_SINGLE_STEP and "
                << "EVD_BC_SEGMENTED_TILE_MULTI_STEP are mutually exclusive"
                << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  if (segmentedTileSuffixBcast &&
      (!segmentedTileWave || !segmentedTileRangeBcast || segmentedTileWaveFullBcast)) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_BC_SEGMENTED_TILE_SUFFIX_BCAST=1 requires tile-wave "
                << "range broadcast mode: EVD_BC_SEGMENTED_TILE_WAVE=1, "
                << "EVD_BC_SEGMENTED_TILE_RANGE_BCAST=1, and "
                << "EVD_BC_SEGMENTED_TILE_FULL_BCAST=0" << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  if (segmentedTileOwnerHandoff &&
      (!segmentedTileWave || !segmentedTileBatchMajor || !segmentedTileSweepOwner ||
       segmentedTileRangeBcast || segmentedTileWaveFullBcast)) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_BC_SEGMENTED_TILE_OWNER_HANDOFF=1 requires batch-major "
                << "sweep-owner tile-wave mode with range/full broadcasts disabled"
                << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  if (segmentedTileSuffixBcast && ctx.size > 1 && ctx.sbrSuffixNccls.empty()) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_BC_SEGMENTED_TILE_SUFFIX_BCAST=1 requires "
                << "initialized suffix NCCL communicators"
                << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }

  EVD_CUDA_CHECK(cudaMemset(dUPacked, 0, sizeof(double) * static_cast<size_t>(packedUElems)));

  const std::vector<long> &hPackedUOffsets = packedUOffsets.host;
  long *dPackedUOffsets = packedUOffsets.device;
  const int packedUSweepCount = static_cast<int>(hPackedUOffsets.size());

  int *com = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&com, sizeof(int) * ctx.n));
  std::vector<int> hCom(static_cast<size_t>(ctx.n));
  for (long sweep = 0; sweep < ctx.n; ++sweep) {
    hCom[static_cast<size_t>(sweep)] =
        (sweep + 1 < ctx.n) ? static_cast<int>(sweep + 1) : static_cast<int>(ctx.n + 3 * ctx.b);
  }

  int dev = 0;
  EVD_CUDA_CHECK(cudaGetDevice(&dev));
  int numBlocksPerSm = 0;
  int numThreads = 32 * 32;
  cudaDeviceProp deviceProp;
  EVD_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  EVD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                      chaseBulgesSegmentPackedKernel<kBcBandwidth>,
                                                      numThreads,
                                                      0));
  int blockNum = numBlocksPerSm * deviceProp.multiProcessorCount;
  int nInt = static_cast<int>(ctx.n);
  int bInt = static_cast<int>(ctx.b);
  int ldBandInt = static_cast<int>(2 * ctx.b);
  if (segmentedFlatStop && envIntOrDefault("EVD_ALLOW_UNSAFE_BC_FLAT_STOP", 0) == 0) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_BC_SEGMENTED_FLAT_STOP=1 is a diagnostic flat-frontier mode. "
                << "It can exercise pause/resume and brown handoff plumbing, but it is not "
                << "full-reference equivalent to the slanted BC wavefront. Set "
                << "EVD_ALLOW_UNSAFE_BC_FLAT_STOP=1 only for kernel diagnostics."
                << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  if (segmentedBrownOnly && envIntOrDefault("EVD_ALLOW_UNSAFE_BC_BROWN_ONLY", 0) == 0) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_BC_SEGMENTED_BROWN_ONLY=1 uses the current full-sweep BC kernel "
                << "with a dependency-safe slanted handoff window. It is diagnostic and "
                << "can expand to full-band communication; set "
                << "EVD_ALLOW_UNSAFE_BC_BROWN_ONLY=1 to run it" << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }

  std::vector<PackedUSlicePlan> cumulativePackedUPlans;
  std::vector<PackedUSlicePlan> blockLocalPackedUPlans;
  std::vector<PackedUSlicePlan> tileSweepOwnerPackedUPlans;
  std::vector<PackedUProgressPlan> tileRowOwnerPackedUPlans;
  std::vector<PackedUProgressPlan> tileProgressPackedUPlans;
  PackedUProgressDeviceWorkspace packedUProgressWorkspace;
  long maxCumulativePackedUPlanElems = 0;
  if (segmentedPackedUChain && segmentedPackedUCompactChain) {
    cumulativePackedUPlans.reserve(ctx.size);
    for (int owner = 0; owner < ctx.size; ++owner) {
      const long segmentEnd = ctx.displs[owner] + ctx.counts[owner];
      const long ownerSweepEnd = std::min<long>(ctx.n - 2, std::max<long>(0, segmentEnd - 1));
      cumulativePackedUPlans.push_back(
          buildPackedUOwnerSlicePlan(ctx, hPackedUOffsets, 0, ownerSweepEnd));
      maxCumulativePackedUPlanElems =
          std::max(maxCumulativePackedUPlanElems, cumulativePackedUPlans.back().totalElems);
    }
  }
  if (segmentedBlockLocal) {
    blockLocalPackedUPlans.reserve(ctx.size);
    for (int owner = 0; owner < ctx.size; ++owner) {
      const long blockStart = ctx.displs[owner];
      const long blockBaseEnd = ctx.displs[owner] + ctx.counts[owner];
      const long blockEnd = (owner + 1 < ctx.size)
                                ? std::min<long>(ctx.n, blockBaseEnd + brownCols)
                                : ctx.n;
      const long localN = std::max<long>(0, blockEnd - blockStart);
      const long localSweepEnd =
          std::min<long>(std::max<long>(0, localN - 2),
                         std::max<long>(0, std::min<long>(ctx.counts[owner],
                                                          ctx.n - 2 - blockStart)));
      blockLocalPackedUPlans.push_back(
          buildPackedUOwnerSlicePlan(ctx,
                                     hPackedUOffsets,
                                     blockStart,
                                     blockStart + localSweepEnd));
      maxCumulativePackedUPlanElems =
          std::max(maxCumulativePackedUPlanElems, blockLocalPackedUPlans.back().totalElems);
    }
  }
  if (segmentedTileWave && segmentedTileSweepOwner) {
    tileSweepOwnerPackedUPlans.reserve(ctx.size);
    long tilePlanBatchSweeps =
        (requestedBatchSweeps > 0) ? requestedBatchSweeps : std::max<long>(1, brownCapacitySweeps);
    if (tilePlanBatchSweeps < 1) {
      tilePlanBatchSweeps = 1;
    }
    const long totalSweepsForPlan = std::max<long>(0, ctx.n - 2);
    for (int owner = 0; owner < ctx.size; ++owner) {
      const long ownerBlockStart = std::min<long>(totalSweepsForPlan, ctx.displs[owner]);
      const long ownerBlockEnd =
          std::min<long>(totalSweepsForPlan, ctx.displs[owner] + ctx.counts[owner]);
      const long ownerSweepStart =
          std::min<long>(totalSweepsForPlan,
                         ceilDiv(ownerBlockStart, tilePlanBatchSweeps) * tilePlanBatchSweeps);
      const long ownerSweepEnd =
          std::min<long>(totalSweepsForPlan,
                         ceilDiv(ownerBlockEnd, tilePlanBatchSweeps) * tilePlanBatchSweeps);
      tileSweepOwnerPackedUPlans.push_back(
          buildPackedUOwnerSlicePlan(ctx,
                                     hPackedUOffsets,
                                     ownerSweepStart,
                                     ownerSweepEnd));
      maxCumulativePackedUPlanElems =
          std::max(maxCumulativePackedUPlanElems,
                   tileSweepOwnerPackedUPlans.back().totalElems);
    }
  }
  if (segmentedTileWave && !segmentedTileSweepOwner && segmentedTileRangeHandoff &&
      segmentedTileRowOwnerPackedUBcast) {
    tileRowOwnerPackedUPlans.reserve(ctx.size);
    for (int owner = 0; owner < ctx.size; ++owner) {
      const long ownerStart = ctx.displs[owner];
      const long ownerEnd = ctx.displs[owner] + ctx.counts[owner];
      tileRowOwnerPackedUPlans.push_back(
          buildPackedURowOwnerPlan(ctx, ownerStart, ownerEnd));
      maxCumulativePackedUPlanElems =
          std::max(maxCumulativePackedUPlanElems,
                   tileRowOwnerPackedUPlans.back().totalElems);
    }
  }
  if (segmentedOwnerSerialPackedUOwnerBcast) {
    maxCumulativePackedUPlanElems = std::max(maxCumulativePackedUPlanElems, packedUElems);
  }
  double *dPackedUCompactChain = nullptr;
  if (maxCumulativePackedUPlanElems > 0) {
    EVD_CUDA_CHECK(evdMalloc(&dPackedUCompactChain,
                     sizeof(double) * static_cast<size_t>(maxCumulativePackedUPlanElems)));
  }
  double *dBandBeforeOwner = nullptr;
  if (debugChangedColumns) {
    EVD_CUDA_CHECK(evdMalloc(&dBandBeforeOwner,
                     sizeof(double) * static_cast<size_t>(2 * ctx.b) *
                         static_cast<size_t>(ctx.n)));
  }

  unsigned long long bandBroadcastDoubles = 0;
  unsigned long long bandSuffixBroadcastDoubles = 0;
  unsigned long long bandOwnerHandoffDoubles = 0;
  unsigned long long bandNeighborDoubles = 0;
  unsigned long long bandNeighborBackDoubles = 0;
  unsigned long long bandWrapGatherDoubles = 0;
  unsigned long long bandPullHandoffDoubles = 0;
  unsigned long long bandPreownerHandoffDoubles = 0;
  unsigned long long bandPullCallCount = 0;
  unsigned long long bandPullTransferCount = 0;
  unsigned long long bandPullMaxTransfersPerCall = 0;
  long maxBandNeighborCols = 0;
  int fullBandNeighborWindows = 0;
  unsigned long long comBroadcastInts = 0;
  unsigned long long packedUNeighborDoubles = 0;
  unsigned long long packedUBroadcastDoubles = 0;
  unsigned long long packedUAllreduceDoubles = 0;
  unsigned long long packedUProgressBroadcastCalls = 0;
  unsigned long long packedUProgressSegmentCount = 0;
  unsigned long long packedUProgressMaxSegments = 0;
  unsigned long long packedUProgressMaxDoubles = 0;
  auto recordPackedUProgressPlan = [&](const PackedUProgressPlan &plan) {
    if (plan.totalElems <= 0 || plan.sweeps.empty()) {
      return;
    }
    packedUProgressBroadcastCalls++;
    const unsigned long long segmentCount =
        static_cast<unsigned long long>(plan.sweeps.size());
    packedUProgressSegmentCount += segmentCount;
    packedUProgressMaxSegments =
        std::max(packedUProgressMaxSegments, segmentCount);
    packedUProgressMaxDoubles =
        std::max<unsigned long long>(
            packedUProgressMaxDoubles,
            static_cast<unsigned long long>(plan.totalElems));
  };
  unsigned long long comOwnerHandoffInts = 0;

  // Each implementation fragment is one verified execution mode. They are
  // kept in the original lexical scope so this structural split does not
  // change synchronization, buffer ownership, or experimental tuning paths.
#include "stages/bc/segmented/block_local.inc"
#include "stages/bc/segmented/row_pipeline.inc"
#include "stages/bc/segmented/tile_wave.inc"
#include "stages/bc/segmented/rolling_wavefront.inc"
#include "stages/bc/segmented/owner_serial.inc"
#include "stages/bc/segmented/finalize.inc"

