void allReducePackedUInChunks(const DistContext &ctx, double *dUPacked, long packedUElems)
{
  const long chunkElems = 32L * 1024L * 1024L;
  double *dTmp = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dTmp, sizeof(double) * static_cast<size_t>(std::min<long>(chunkElems, packedUElems))));
  for (long offset = 0; offset < packedUElems; offset += chunkElems) {
    const long count = std::min<long>(chunkElems, packedUElems - offset);
    NCCL_CHECK_LOCAL(ncclAllReduce(dUPacked + offset,
                                   dTmp,
                                   static_cast<size_t>(count),
                                   ncclDouble,
                                   ncclSum,
                                   ctx.nccl,
                                   ctx.commStream));
    EVD_CUDA_CHECK(cudaMemcpyAsync(dUPacked + offset,
                          dTmp,
                          sizeof(double) * static_cast<size_t>(count),
                          cudaMemcpyDeviceToDevice,
                          ctx.commStream));
    EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
  }
  EVD_CUDA_CHECK(evdFree(dTmp));
}

void runBcPipelineLocalRangesPackedU(const DistContext &ctx,
                                     double *dBand,
                                     double *dUPacked,
                                     long packedUElems,
                                     const PackedUOffsetTable &packedUOffsets)
{
  if (ctx.b != kBcBandwidth) {
    std::cerr << "EVD currently instantiates the existing BC kernel for b=32; got b="
              << ctx.b << std::endl;
    MPI_Abort(ctx.comm, 1);
  }

  const long totalSweeps = std::max<long>(0, ctx.n - 2);
  std::vector<long> sweepCounts;
  std::vector<long> sweepDispls;
  buildSweepDistributionFromColumnBlocks(totalSweeps,
                                         ctx.counts,
                                         ctx.displs,
                                         &sweepCounts,
                                         &sweepDispls);
  if (ctx.rank == 0) {
    std::cout << "BC pipeline-local sweep ranges:";
    for (int r = 0; r < ctx.size; ++r) {
      std::cout << " r" << r << "=[" << sweepDispls[r]
                << "," << (sweepDispls[r] + sweepCounts[r]) << ")";
    }
    std::cout << std::endl;
  }

  int *com = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&com, sizeof(int) * ctx.n));
  EVD_CUDA_CHECK(cudaMemset(com, 0, sizeof(int) * ctx.n));
  const long sweepStart = sweepDispls[ctx.rank];
  const long sweepEnd = sweepStart + sweepCounts[ctx.rank];
  if (sweepStart > 0) {
    const int complete = static_cast<int>(ctx.n + 3 * ctx.b);
    EVD_CUDA_CHECK(cudaMemcpy(com + sweepStart - 1, &complete, sizeof(int), cudaMemcpyHostToDevice));
  }
  EVD_CUDA_CHECK(cudaMemset(dUPacked, 0, sizeof(double) * static_cast<size_t>(packedUElems)));

  const std::vector<long> &hPackedUOffsets = packedUOffsets.host;
  long *dPackedUOffsets = packedUOffsets.device;
  const bool compactPackedUComm = envIntOrDefault("EVD_BC_PACKED_U_COMPACT", 0) != 0;
  std::vector<PackedUSlicePlan> packedUPlans;
  long maxPackedUPlanElems = 0;
  if (compactPackedUComm) {
    packedUPlans.reserve(ctx.size);
    for (int owner = 0; owner < ctx.size; ++owner) {
      const long ownerStart = sweepDispls[owner];
      const long ownerEnd = ownerStart + sweepCounts[owner];
      packedUPlans.push_back(buildPackedUOwnerSlicePlan(ctx, hPackedUOffsets, ownerStart, ownerEnd));
      maxPackedUPlanElems = std::max(maxPackedUPlanElems, packedUPlans.back().totalElems);
    }
  }
  double *dPackedUCompact = nullptr;
  if (maxPackedUPlanElems > 0) {
    EVD_CUDA_CHECK(evdMalloc(&dPackedUCompact, sizeof(double) * static_cast<size_t>(maxPackedUPlanElems)));
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
  const int packedUSweepCount = static_cast<int>(hPackedUOffsets.size());
  const bool suffixBandHandoff =
      envIntOrDefault("EVD_BC_PIPELINE_LOCAL_SUFFIX_HANDOFF", 1) != 0;
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

  if (ctx.rank > 0) {
    const long recvStart = suffixBandHandoff ? sweepStart : 0;
    handoffBandRangeToNextOwner(ctx, dBand, recvStart, ctx.n, ctx.rank - 1);
  }

  if (sweepEnd > sweepStart) {
    int zero = 0;
    EVD_CUDA_CHECK(cudaMemcpyToSymbol(bcStopFlag, &zero, sizeof(int)));
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(blockNum, 1, 1);
    EVD_CUDA_CHECK(cudaLaunchCooperativeKernel((void *)chaseBulgesRangePackedKernel<kBcBandwidth>,
                                      dimGrid,
                                      dimBlock,
                                      kernelArgs));
    EVD_CUDA_CHECK(cudaGetLastError());
    EVD_CUDA_CHECK(cudaDeviceSynchronize());
  }

  if (ctx.rank + 1 < ctx.size) {
    const long sendStart = suffixBandHandoff ? sweepEnd : 0;
    handoffBandRangeToNextOwner(ctx, dBand, sendStart, ctx.n, ctx.rank);
  }

  MPI_CHECK(MPI_Barrier(ctx.comm));
  unsigned long long packedUBroadcastDoubles = 0;
  const long lastSweepUCount =
      ctx.n - (static_cast<long>(packedUSweepCount - 1) * kBcBackSweepRows + 1) - 1;
  for (int owner = 0; owner < ctx.size; ++owner) {
    const long ownerStart = sweepDispls[owner];
    const long ownerEnd = ownerStart + sweepCounts[owner];
    for (int sweepIndex = 0; sweepIndex < packedUSweepCount; ++sweepIndex) {
      const long totalU =
          lastSweepUCount + static_cast<long>(sweepIndex) * kBcBackSweepRows;
      const long sliceStart = std::max<long>(0, ownerStart);
      const long sliceEnd = std::min<long>(ownerEnd, totalU);
      if (sliceEnd > sliceStart) {
        packedUBroadcastDoubles += static_cast<unsigned long long>(sliceEnd - sliceStart) *
                                   static_cast<unsigned long long>(kBcBackSweepRows);
      }
    }
    if (compactPackedUComm) {
      broadcastPackedUOwnerSlicesCompact(ctx,
                                         dUPacked,
                                         dPackedUCompact,
                                         packedUPlans[owner],
                                         owner);
    } else {
      broadcastPackedUOwnerSlices(ctx, dUPacked, hPackedUOffsets, ownerStart, ownerEnd, owner);
    }
  }
  if (ctx.rank == 0) {
    unsigned long long bandNeighborDoubles = 0;
    for (int owner = 0; owner + 1 < ctx.size; ++owner) {
      const long sendStart = suffixBandHandoff ? (sweepDispls[owner] + sweepCounts[owner]) : 0;
      const long sendCols = std::max<long>(0, ctx.n - sendStart);
      bandNeighborDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                             static_cast<unsigned long long>(sendCols);
    }
    std::cout << "BC pipeline-local comm estimate: band_neighbor="
              << static_cast<double>(bandNeighborDoubles * sizeof(double)) / (1024.0 * 1024.0)
              << " MiB, packedU_bcast="
              << static_cast<double>(packedUBroadcastDoubles * sizeof(double)) /
                     (1024.0 * 1024.0 * 1024.0)
              << " GiB, packed_u_compact=" << (compactPackedUComm ? 1 : 0)
              << ", packed_u_compact_max="
              << static_cast<double>(maxPackedUPlanElems * sizeof(double)) /
                     (1024.0 * 1024.0 * 1024.0)
              << " GiB, suffix_band_handoff=" << (suffixBandHandoff ? 1 : 0)
              << std::endl;
  }

  if (dPackedUCompact != nullptr) {
    EVD_CUDA_CHECK(evdFree(dPackedUCompact));
  }
  EVD_CUDA_CHECK(evdFree(com));
}

