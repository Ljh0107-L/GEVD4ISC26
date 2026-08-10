  if (segmentedBlockLocal) {
    long maxSegmentRetries =
        static_cast<long>(envIntOrDefault("EVD_BC_SEGMENTED_MAX_RETRIES", 4096));
    if (maxSegmentRetries < 1) {
      maxSegmentRetries = 1;
    }
    if (ctx.rank == 0) {
      std::cout << "BC segmented block-local brown_cols=" << brownCols
                << " full_bcast=" << (segmentedBlockLocalFullBcast ? 1 : 0)
                << " slanted=" << (segmentedBlockLocalSlanted ? 1 : 0)
                << " single_step=" << (segmentedBlockLocalSingleStep ? 1 : 0)
                << std::endl;
    }
    for (int owner = 0; owner < ctx.size; ++owner) {
      const long blockStart = ctx.displs[owner];
      const long blockEnd = ctx.displs[owner] + ctx.counts[owner];
      const long segmentStop =
          (owner + 1 < ctx.size) ? std::min<long>(ctx.n, blockEnd + brownCols) : ctx.n;
      const long sweepEnd =
          (owner + 1 < ctx.size)
              ? std::min<long>(std::max<long>(0, ctx.n - 2),
                               std::max<long>(0, blockEnd - 1))
              : std::max<long>(0, ctx.n - 2);
      if (ctx.rank == 0) {
        std::cout << "BC segmented block-local owner " << owner
                  << " block=[" << blockStart << "," << blockEnd << ")"
                  << " stop=" << segmentStop
                  << " sweeps=[0," << sweepEnd << ")"
                  << std::endl;
      }

      std::vector<int> hComBeforeOwner = hCom;
      if (ctx.rank == owner && sweepEnd > 0) {
        dim3 dimBlock(32, 32, 1);
        dim3 dimGrid(blockNum, 1, 1);
        long blockLocalBatchSweeps =
            (requestedBatchSweeps > 0) ? requestedBatchSweeps : brownCapacitySweeps;
        blockLocalBatchSweeps = std::max<long>(
            1, std::min<long>(blockLocalBatchSweeps, brownCapacitySweeps));
        for (long batchStart = 0; batchStart < sweepEnd; batchStart += blockLocalBatchSweeps) {
          const long batchEnd = std::min<long>(sweepEnd, batchStart + blockLocalBatchSweeps);
          EVD_CUDA_CHECK(cudaMemcpy(com, hCom.data(), sizeof(int) * hCom.size(), cudaMemcpyHostToDevice));
          int sweepStartInt = static_cast<int>(batchStart);
          int sweepEndInt = static_cast<int>(batchEnd);
	          int segmentStartInt =
	              segmentedBlockLocalSingleStep ? -3
	                                            : (segmentedBlockLocalSlanted ? -6 : -5);
          int segmentEndInt = static_cast<int>(segmentStop);
          int segmentStopSweepInt =
              (!segmentedBlockLocalSingleStep && segmentedBlockLocalSlanted)
                  ? static_cast<int>(batchEnd - 1)
                  : -1;
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
              (void *)&segmentStartInt,
              (void *)&segmentEndInt,
              (void *)&segmentStopSweepInt,
              (void *)&blockNum,
              (void *)&com,
          };
          long retry = 0;
          bool batchReached = false;
          long long prevProgressScore = -1;
          do {
            int zero = 0;
            EVD_CUDA_CHECK(cudaMemcpyToSymbol(bcStopFlag, &zero, sizeof(int)));
            EVD_CUDA_CHECK(cudaLaunchCooperativeKernel((void *)chaseBulgesSegmentPackedKernel<kBcBandwidth>,
                                              dimGrid,
                                              dimBlock,
                                              kernelArgs));
            EVD_CUDA_CHECK(cudaGetLastError());
            EVD_CUDA_CHECK(cudaDeviceSynchronize());
            EVD_CUDA_CHECK(cudaMemcpy(hCom.data(), com, sizeof(int) * hCom.size(), cudaMemcpyDeviceToHost));
            long reached = 0;
            long long progressScore = 0;
            for (long sweep = batchStart; sweep < batchEnd; ++sweep) {
              const long sweepLag =
                  segmentedBlockLocalSlanted ? std::max<long>(0, (batchEnd - 1) - sweep) : 0;
              const long target =
                  std::min<long>(ctx.n, segmentStop + 2 * ctx.b * sweepLag);
              const long progress = hCom[static_cast<size_t>(sweep)];
              if (progress >= target || progress >= ctx.n + 3 * ctx.b) {
                reached++;
              }
              progressScore += std::min<long>(progress, target);
            }
            batchReached = (reached == (batchEnd - batchStart));
            if (batchReached || progressScore == prevProgressScore) {
              break;
            }
            prevProgressScore = progressScore;
            retry++;
          } while (retry < maxSegmentRetries);
          if (!batchReached && printSegmentedBatch) {
            std::cout << "BC segmented block-local owner " << owner
                      << " batch=[" << batchStart << "," << batchEnd << ")"
                      << " stopped before segment_stop=" << segmentStop
                      << " retries=" << (retry + 1) << std::endl;
          }
        }
        long progressMin = ctx.n + 3 * ctx.b;
        long progressMax = 0;
        long completed = 0;
        long reachedStop = 0;
        long firstBelowStop = -1;
        for (long sweep = 0; sweep < sweepEnd; ++sweep) {
          const long progress = hCom[static_cast<size_t>(sweep)];
          progressMin = std::min(progressMin, progress);
          progressMax = std::max(progressMax, progress);
          if (progress >= ctx.n + 3 * ctx.b) {
            completed++;
          }
          if (progress >= segmentStop || progress >= ctx.n + 3 * ctx.b) {
            reachedStop++;
          } else if (firstBelowStop < 0) {
            firstBelowStop = sweep;
          }
        }
        std::cout << "BC segmented block-local progress owner " << owner
                  << " active_sweeps=" << sweepEnd
                  << " completed=" << completed
                  << " reached_stop=" << reachedStop
                  << " progress_min=" << progressMin
                  << " progress_max=" << progressMax
                  << " first_below_stop=" << firstBelowStop
                  << std::endl;
      }

      comBroadcastInts += static_cast<unsigned long long>(hCom.size());
      MPI_CHECK(MPI_Bcast(hCom.data(),
                          static_cast<int>(hCom.size()),
                          MPI_INT,
                          owner,
                          ctx.comm));

      PackedUProgressPlan progressPlan = buildPackedUProgressPlan(ctx, hComBeforeOwner, hCom);
      if (progressPlan.totalElems > 0) {
        if (dPackedUCompactChain == nullptr ||
            progressPlan.totalElems > maxCumulativePackedUPlanElems) {
          if (dPackedUCompactChain != nullptr) {
            EVD_CUDA_CHECK(evdFree(dPackedUCompactChain));
          }
          EVD_CUDA_CHECK(evdMalloc(&dPackedUCompactChain,
                           sizeof(double) * static_cast<size_t>(progressPlan.totalElems)));
          maxCumulativePackedUPlanElems = progressPlan.totalElems;
        }
        packedUBroadcastDoubles += static_cast<unsigned long long>(progressPlan.totalElems);
        recordPackedUProgressPlan(progressPlan);
        broadcastPackedUProgressCompact(ctx,
                                        dUPacked,
                                        dPackedUCompactChain,
                                      progressPlan,
                                      dPackedUOffsets,
                                      packedUSweepCount,
                                      owner,
                                      &packedUProgressWorkspace);
      }

      if (segmentedBlockLocalFullBcast) {
        bandBroadcastDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                                static_cast<unsigned long long>(ctx.n);
        broadcastBandSuffix(ctx, dBand, 0, owner);
      } else if (owner + 1 < ctx.size) {
        const long handoffStart = std::max<long>(0, blockEnd - ctx.b);
        const long handoffEnd = std::min<long>(ctx.n, segmentStop + ctx.b);
        const long cols = std::max<long>(0, handoffEnd - handoffStart);
        maxBandNeighborCols = std::max(maxBandNeighborCols, cols);
        if (cols > 0) {
          bandNeighborDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                                  static_cast<unsigned long long>(cols);
        }
        handoffBandRangeToNextOwner(ctx, dBand, handoffStart, handoffEnd, owner);
      }
    }
    if (segmentedOwnerGatherBack && !segmentedBlockLocalFullBcast && ctx.size > 1) {
      if (!allCountsEqual(ctx.counts)) {
        if (ctx.rank == 0) {
          std::cerr << "BC segmented block-local owner-gather-back currently requires "
                    << "equal rank column blocks" << std::endl;
        }
        MPI_Abort(ctx.comm, 2);
      }
      const long tileRows = ctx.counts.empty() ? ctx.n : ctx.counts[0];
      const long tileCount = static_cast<long>(ctx.size);
      const long haloCols = brownCols + ctx.b;
      for (int target = 0; target < ctx.size; ++target) {
        const long ownedStart = ctx.displs[target];
        const long ownedEnd = ctx.displs[target] + ctx.counts[target];
        bandOwnerHandoffDoubles +=
            handoffRowTileWrapRangeToOwner(ctx,
                                           dBand,
                                           ownedStart,
                                           ownedEnd,
                                           tileRows,
                                           tileCount,
                                           haloCols,
                                           target,
                                           &maxBandNeighborCols);
      }
    }
