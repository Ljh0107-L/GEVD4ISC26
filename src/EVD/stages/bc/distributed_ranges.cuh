long bcBandSyncStart(long n, long ownerStart, long ownerEnd, int bandSyncMode)
{
  if (bandSyncMode == 0) {
    return 0;
  }
  if (bandSyncMode == 1) {
    return std::max<long>(0, std::min<long>(n, ownerStart));
  }
  if (bandSyncMode == 2) {
    return std::max<long>(0, std::min<long>(n, ownerEnd));
  }
  if (bandSyncMode == 3) {
    return std::max<long>(0, std::min<long>(n, ownerEnd));
  }
  return -1;
}

long bcBandSyncEnd(long n, long syncStart, int bandSyncMode, long brownCols)
{
  if (bandSyncMode == 3) {
    return std::max<long>(syncStart, std::min<long>(n, syncStart + brownCols));
  }
  return n;
}

const char *bcBandSyncModeName(int bandSyncMode)
{
  switch (bandSyncMode) {
    case 0:
      return "full-band";
    case 1:
      return "owner-start-suffix";
    case 2:
      return "next-owner-start-suffix";
    case 3:
      return "owner-boundary-window";
    default:
      return "invalid";
  }
}

void runBcDistributedRangesPackedU(const DistContext &ctx,
                                   double *dBand,
                                   double *dUPacked,
                                   long packedUElems,
                                   int bandSyncMode,
                                   const PackedUOffsetTable &packedUOffsets,
                                   const std::vector<int> *initialProgress = nullptr,
                                   bool preservePackedU = false)
{
  if (ctx.b != kBcBandwidth) {
    std::cerr << "EVD currently instantiates the existing BC kernel for b=32; got b="
              << ctx.b << std::endl;
    MPI_Abort(ctx.comm, 1);
  }
  if (bcBandSyncStart(ctx.n, 0, 0, bandSyncMode) < 0) {
    if (ctx.rank == 0) {
      std::cerr << "Invalid EVD_BC_SYNC_MODE=" << bandSyncMode
                << "; supported values are 0=full-band, 1=owner-start-suffix, 2=next-owner-start-suffix, 3=owner-boundary-window" << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  if (bandSyncMode == 3 && envIntOrDefault("EVD_ALLOW_UNSAFE_BC_BROWN_WINDOW", 0) == 0) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_BC_SYNC_MODE=3 is an experimental brown-window diagnostic; set "
                << "EVD_ALLOW_UNSAFE_BC_BROWN_WINDOW=1 to run it" << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  long brownCols = static_cast<long>(envIntOrDefault("EVD_BC_BROWN_COLS", static_cast<int>(4 * ctx.b)));
  if (brownCols <= 0) {
    if (ctx.rank == 0) {
      std::cerr << "EVD_BC_BROWN_COLS must be positive; got " << brownCols << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  const bool bandNeighborHandoff = envIntOrDefault("EVD_BC_BAND_NEIGHBOR", 0) != 0;
  const bool compactPackedUComm = envIntOrDefault("EVD_BC_PACKED_U_COMPACT", 0) != 0;

  const long totalSweeps = std::max<long>(0, ctx.n - 2);
  std::vector<long> sweepCounts;
  std::vector<long> sweepDispls;
  buildSweepDistributionFromColumnBlocks(totalSweeps,
                                         ctx.counts,
                                         ctx.displs,
                                         &sweepCounts,
                                         &sweepDispls);

  int *com = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&com, sizeof(int) * ctx.n));
  if (initialProgress != nullptr) {
    if (initialProgress->size() != static_cast<size_t>(ctx.n)) {
      if (ctx.rank == 0) {
        std::cerr << "BC resume progress has " << initialProgress->size()
                  << " entries; expected " << ctx.n << std::endl;
      }
      MPI_Abort(ctx.comm, 2);
    }
    EVD_CUDA_CHECK(cudaMemcpy(com,
                     initialProgress->data(),
                     sizeof(int) * initialProgress->size(),
                     cudaMemcpyHostToDevice));
  } else {
    EVD_CUDA_CHECK(cudaMemset(com, 0, sizeof(int) * ctx.n));
  }
  const long sweepStart = sweepDispls[ctx.rank];
  const long sweepEnd = sweepStart + sweepCounts[ctx.rank];
  if (sweepStart > 0) {
    const int complete = static_cast<int>(ctx.n + 3 * ctx.b);
    EVD_CUDA_CHECK(cudaMemcpy(com + sweepStart - 1, &complete, sizeof(int), cudaMemcpyHostToDevice));
  }
  if (!preservePackedU) {
    EVD_CUDA_CHECK(cudaMemset(dUPacked, 0, sizeof(double) * static_cast<size_t>(packedUElems)));
  }

  const std::vector<long> &hPackedUOffsets = packedUOffsets.host;
  long *dPackedUOffsets = packedUOffsets.device;
  const int packedUSweepCount = static_cast<int>(hPackedUOffsets.size());
  const long lastSweepUCount = ctx.n - (static_cast<long>(packedUSweepCount - 1) * kBcBackSweepRows + 1) - 1;
  std::vector<PackedUSlicePlan> packedUPlans;
  long maxPackedUPlanElems = 0;
  if (compactPackedUComm) {
    packedUPlans.reserve(ctx.size);
    for (int owner = 0; owner < ctx.size; ++owner) {
      const long ownerStart = std::min<long>(sweepDispls[owner], ctx.n);
      const long ownerEnd = std::min<long>(sweepDispls[owner] + sweepCounts[owner], ctx.n);
      packedUPlans.push_back(buildPackedUOwnerSlicePlan(ctx, hPackedUOffsets, ownerStart, ownerEnd));
      maxPackedUPlanElems = std::max(maxPackedUPlanElems, packedUPlans.back().totalElems);
    }
  }
  double *dPackedUCompact = nullptr;
  if (maxPackedUPlanElems > 0) {
    EVD_CUDA_CHECK(evdMalloc(&dPackedUCompact, sizeof(double) * static_cast<size_t>(maxPackedUPlanElems)));
  }
  if (ctx.rank == 0 && envIntOrDefault("EVD_DEBUG_STAGE_PROGRESS", 0) != 0) {
    std::cout << (initialProgress == nullptr ? "BC sweep ranges:"
                                               : "BC resumed sweep ranges:");
    for (int r = 0; r < ctx.size; ++r) {
      std::cout << " r" << r << "=[" << sweepDispls[r]
                << "," << (sweepDispls[r] + sweepCounts[r]) << ")";
    }
    std::cout << std::endl;
  }

  int dev = 0;
  EVD_CUDA_CHECK(cudaGetDevice(&dev));
  int numBlocksPerSm = 0;
  int numThreads = 32 * 32;
  cudaDeviceProp deviceProp;
  EVD_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  EVD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                      chaseBulgesRangePackedKernel<kBcBandwidth>,
                                                      numThreads,
                                                      0));
  int blockNum = numBlocksPerSm * deviceProp.multiProcessorCount;
  int nInt = static_cast<int>(ctx.n);
  int bInt = static_cast<int>(ctx.b);
  int ldBandInt = static_cast<int>(2 * ctx.b);
  int sweepStartInt = static_cast<int>(sweepStart);
  int sweepEndInt = static_cast<int>(sweepEnd);
  void *kernelArgs[] = {
      (void *)&nInt,
      (void *)&bInt,
      (void *)&dBand,
      (void *)&ldBandInt,
      (void *)&dUPacked,
      (void *)&dPackedUOffsets,
      (void *)&packedUSweepCount,
      (void *)&sweepStartInt,
      (void *)&sweepEndInt,
      (void *)&blockNum,
      (void *)&com,
  };

  unsigned long long bandBroadcastDoubles = 0;
  unsigned long long bandNeighborDoubles = 0;
  unsigned long long packedUBroadcastDoubles = 0;
  for (int owner = 0; owner < ctx.size; ++owner) {
    if (ctx.rank == owner && sweepCounts[owner] > 0) {
      int zero = 0;
      EVD_CUDA_CHECK(cudaMemcpyToSymbol(bcStopFlag, &zero, sizeof(int)));
      sweepStartInt = static_cast<int>(sweepDispls[owner]);
      sweepEndInt = static_cast<int>(sweepDispls[owner] + sweepCounts[owner]);
      dim3 dimBlock(32, 32, 1);
      dim3 dimGrid(blockNum, 1, 1);
      if (initialProgress != nullptr && owner == 0) {
        int segmentStartInt = -2;
        int segmentEndInt = static_cast<int>(ctx.n);
        int segmentStopSweepInt = -1;
        void *resumeArguments[] = {
            (void *)&nInt,
            (void *)&bInt,
            (void *)&dBand,
            (void *)&ldBandInt,
            (void *)&dUPacked,
            (void *)&dPackedUOffsets,
            (void *)&packedUSweepCount,
            (void *)&sweepStartInt,
            (void *)&sweepEndInt,
            (void *)&segmentStartInt,
            (void *)&segmentEndInt,
            (void *)&segmentStopSweepInt,
            (void *)&blockNum,
            (void *)&com,
        };
        EVD_CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void *)chaseBulgesSegmentPackedKernel<kBcBandwidth>,
            dimGrid,
            dimBlock,
            resumeArguments));
      } else {
        EVD_CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void *)chaseBulgesRangePackedKernel<kBcBandwidth>,
            dimGrid,
            dimBlock,
            kernelArgs));
      }
      EVD_CUDA_CHECK(cudaGetLastError());
      EVD_CUDA_CHECK(cudaDeviceSynchronize());
    }

    const long ownerStart = sweepDispls[owner];
    const long ownerEnd = sweepDispls[owner] + sweepCounts[owner];
    const long bandSyncStart = bcBandSyncStart(ctx.n, ownerStart, ownerEnd, bandSyncMode);
    const long bandSyncEnd = bcBandSyncEnd(ctx.n, bandSyncStart, bandSyncMode, brownCols);
    if (bandSyncStart < bandSyncEnd) {
      const unsigned long long bandDoubles =
          static_cast<unsigned long long>(2 * ctx.b) *
          static_cast<unsigned long long>(bandSyncEnd - bandSyncStart);
      if (bandNeighborHandoff && owner + 1 < ctx.size) {
        bandNeighborDoubles += bandDoubles;
      } else if (!bandNeighborHandoff) {
        bandBroadcastDoubles += bandDoubles;
      }
    }
    for (int sweepIndex = 0; sweepIndex < packedUSweepCount; ++sweepIndex) {
      const long totalU = lastSweepUCount + static_cast<long>(sweepIndex) * kBcBackSweepRows;
      const long sliceStart = std::max<long>(0, ownerStart);
      const long sliceEnd = std::min<long>(ownerEnd, totalU);
      if (sliceEnd > sliceStart) {
        packedUBroadcastDoubles += static_cast<unsigned long long>(sliceEnd - sliceStart) *
                                   static_cast<unsigned long long>(kBcBackSweepRows);
      }
    }
    if (bandNeighborHandoff) {
      handoffBandRangeToNextOwner(ctx, dBand, bandSyncStart, bandSyncEnd, owner);
    } else if (bandSyncMode == 3) {
      broadcastBandColumnRange(ctx, dBand, bandSyncStart, bandSyncEnd, owner);
    } else {
      broadcastBandSuffix(ctx, dBand, bandSyncStart, owner);
    }
    if (compactPackedUComm) {
      broadcastPackedUOwnerSlicesCompact(ctx,
                                         dUPacked,
                                         dPackedUCompact,
                                         packedUPlans[owner],
                                         owner);
    } else {
      broadcastPackedUOwnerSlices(ctx,
                                  dUPacked,
                                  hPackedUOffsets,
                                  ownerStart,
                                  ownerEnd,
                                  owner);
    }
  }
  if (ctx.rank == 0 && envIntOrDefault("EVD_DEBUG_STAGE_PROGRESS", 0) != 0) {
    std::cout << "BC range comm estimate: band_bcast="
              << static_cast<double>(bandBroadcastDoubles * sizeof(double)) / (1024.0 * 1024.0)
              << " MiB, band_neighbor="
              << static_cast<double>(bandNeighborDoubles * sizeof(double)) / (1024.0 * 1024.0)
              << " MiB, packedU_bcast="
              << static_cast<double>(packedUBroadcastDoubles * sizeof(double)) / (1024.0 * 1024.0 * 1024.0)
              << " GiB, band_sync_mode=" << bcBandSyncModeName(bandSyncMode)
              << ", band_neighbor_handoff=" << (bandNeighborHandoff ? 1 : 0)
              << ", packed_u_compact=" << (compactPackedUComm ? 1 : 0)
              << ", packed_u_compact_max="
              << static_cast<double>(maxPackedUPlanElems * sizeof(double)) / (1024.0 * 1024.0 * 1024.0)
              << " GiB"
              << ", brown_cols=" << brownCols
              << std::endl;
  }

  if (dPackedUCompact != nullptr) {
    EVD_CUDA_CHECK(evdFree(dPackedUCompact));
  }
  EVD_CUDA_CHECK(evdFree(com));
}

