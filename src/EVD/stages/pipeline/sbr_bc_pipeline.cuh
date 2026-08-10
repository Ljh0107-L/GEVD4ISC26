#pragma once

std::vector<int> makeStagePipelineProgress(const DistContext &ctx)
{
  std::vector<int> progress(static_cast<size_t>(ctx.n), 0);
  for (long sweep = 0; sweep < ctx.n; ++sweep) {
    progress[static_cast<size_t>(sweep)] =
        (sweep + 1 < ctx.n)
            ? static_cast<int>(sweep + 1)
            : static_cast<int>(ctx.n + 3 * ctx.b);
  }
  return progress;
}

double *produceInitialBcPrefix(const DistContext &ctx,
                               const StagePipelinePlan &plan,
                               double *dBand,
                               double *dUPacked,
                               int *dProgress,
                               std::vector<int> *progress,
                               const PackedUOffsetTable &packedUOffsets)
{
  if (ctx.rank != 0) {
    return nullptr;
  }

  int device = 0;
  EVD_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties;
  EVD_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  int blocksPerSm = 0;
  EVD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocksPerSm,
      chaseBulgesSegmentPackedKernel<kBcBandwidth>,
      32 * 32,
      0));
  int blockCount = blocksPerSm * properties.multiProcessorCount;

  int n = static_cast<int>(ctx.n);
  int b = static_cast<int>(ctx.b);
  int ldBand = static_cast<int>(2 * ctx.b);
  int packedSweepCount = static_cast<int>(packedUOffsets.host.size());
  int sweepStart = 0;
  int sweepEnd = static_cast<int>(plan.initialSweeps);
  int segmentStart = -4;  // dependency-safe slanted multi-step frontier
  int segmentEnd = static_cast<int>(plan.segmentEnd);
  int segmentStopSweep = sweepEnd - 1;
  long *dOffsets = packedUOffsets.device;
  void *arguments[] = {
      &n,
      &b,
      &dBand,
      &ldBand,
      &dUPacked,
      &dOffsets,
      &packedSweepCount,
      &sweepStart,
      &sweepEnd,
      &segmentStart,
      &segmentEnd,
      &segmentStopSweep,
      &blockCount,
      &dProgress,
  };

  long long previousScore = -1;
  bool reachedFrontier = false;
  for (int retry = 0; retry < 4096; ++retry) {
    int zero = 0;
    EVD_CUDA_CHECK(cudaMemcpyToSymbol(bcStopFlag, &zero, sizeof(int)));
    dim3 threads(32, 32, 1);
    dim3 blocks(blockCount, 1, 1);
    EVD_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(
            chaseBulgesSegmentPackedKernel<kBcBandwidth>),
        blocks,
        threads,
        arguments));
    EVD_CUDA_CHECK(cudaGetLastError());
    EVD_CUDA_CHECK(cudaDeviceSynchronize());
    EVD_CUDA_CHECK(cudaMemcpy(progress->data(),
                     dProgress,
                     sizeof(int) * progress->size(),
                     cudaMemcpyDeviceToHost));

    long reached = 0;
    long long score = 0;
    for (long sweep = 0; sweep < plan.initialSweeps; ++sweep) {
      const long lag = plan.initialSweeps - 1 - sweep;
      const long target =
          std::min<long>(ctx.n, plan.segmentEnd + 2 * ctx.b * lag);
      const long value = (*progress)[static_cast<size_t>(sweep)];
      if (value >= target || value >= ctx.n + 3 * ctx.b) {
        ++reached;
      }
      score += std::min<long>(value, target);
    }
    reachedFrontier = reached == plan.initialSweeps;
    if (reachedFrontier || score == previousScore) {
      break;
    }
    previousScore = score;
  }
  if (!reachedFrontier) {
    std::cerr << "stage-pipeline BC producer stopped before its planned frontier"
              << std::endl;
    MPI_Abort(ctx.comm, 2);
  }

  const size_t prefixElements =
      static_cast<size_t>(2 * ctx.b) *
      static_cast<size_t>(plan.changedBandEnd);
  double *dChangedPrefix = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dChangedPrefix, sizeof(double) * prefixElements));
  EVD_CUDA_CHECK(cudaMemcpy(dChangedPrefix,
                   dBand,
                   sizeof(double) * prefixElements,
                   cudaMemcpyDeviceToDevice));
  return dChangedPrefix;
}

void mergeSbrAndBcStageState(const DistContext &ctx,
                             const StagePipelinePlan &plan,
                             double *dBand,
                             const double *dChangedPrefix,
                             std::vector<int> *progress)
{
  // Suffix ranks reach this point only after their SBR work is complete.
  // The barrier joins the stages without preventing rank zero's BC producer
  // from overlapping that work.
  MPI_CHECK(MPI_Barrier(ctx.comm));

  // The last rank participated in every SBR suffix communicator, so it owns a
  // complete raw band even after rank zero left SBR.
  const size_t fullBandElements =
      static_cast<size_t>(2 * ctx.b) * static_cast<size_t>(ctx.n);
  NCCL_CHECK_LOCAL(ncclBroadcast(dBand,
                                 dBand,
                                 fullBandElements,
                                 ncclDouble,
                                 ctx.size - 1,
                                 ctx.nccl,
                                 ctx.commStream));
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));

  MPI_CHECK(MPI_Bcast(progress->data(),
                      static_cast<int>(progress->size()),
                      MPI_INT,
                      0,
                      ctx.comm));

  // Overlay the prefix already transformed by rank zero.  Columns after this
  // prefix remain the freshly published SBR output.
  const size_t prefixElements =
      static_cast<size_t>(2 * ctx.b) *
      static_cast<size_t>(plan.changedBandEnd);
  const double *prefixSource = (ctx.rank == 0) ? dChangedPrefix : dBand;
  NCCL_CHECK_LOCAL(ncclBroadcast(prefixSource,
                                 dBand,
                                 prefixElements,
                                 ncclDouble,
                                 0,
                                 ctx.nccl,
                                 ctx.commStream));
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
}

void runBcStagePipelinePackedU(const DistContext &ctx,
                               const StagePipelinePlan &plan,
                               double *dBand,
                               double *dUPacked,
                               long packedUElems,
                               const PackedUOffsetTable &packedUOffsets)
{
  if (!plan.enabled) {
    runBcDistributedRangesPackedU(ctx,
                                  dBand,
                                  dUPacked,
                                  packedUElems,
                                  1,
                                  packedUOffsets);
    return;
  }

  EVD_CUDA_CHECK(cudaMemset(dUPacked,
                   0,
                   sizeof(double) * static_cast<size_t>(packedUElems)));

  std::vector<int> progress = makeStagePipelineProgress(ctx);
  int *dProgress = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dProgress, sizeof(int) * static_cast<size_t>(ctx.n)));
  EVD_CUDA_CHECK(cudaMemcpy(dProgress,
                   progress.data(),
                   sizeof(int) * progress.size(),
                   cudaMemcpyHostToDevice));

  double *dChangedPrefix = produceInitialBcPrefix(ctx,
                                                   plan,
                                                   dBand,
                                                   dUPacked,
                                                   dProgress,
                                                   &progress,
                                                   packedUOffsets);
  mergeSbrAndBcStageState(ctx,
                          plan,
                          dBand,
                          dChangedPrefix,
                          &progress);

  if (dChangedPrefix != nullptr) {
    EVD_CUDA_CHECK(evdFree(dChangedPrefix));
  }
  EVD_CUDA_CHECK(evdFree(dProgress));

  // Resume the ordinary coarse distributed BC ranges.  Rank zero resumes
  // directly from progress; no reflector is applied twice.
  runBcDistributedRangesPackedU(ctx,
                                dBand,
                                dUPacked,
                                packedUElems,
                                1,
                                packedUOffsets,
                                &progress,
                                true);
}
