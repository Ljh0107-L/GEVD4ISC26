  } else {
  for (int owner = 0; owner < ctx.size; ++owner) {
    std::vector<int> hComBeforeOwner;
    if (segmentedOwnerSerialPackedUOwnerBcast) {
      hComBeforeOwner = hCom;
    }
    const long segmentStart = ctx.displs[owner];
    const long segmentEnd = ctx.displs[owner] + ctx.counts[owner];
    const bool useBrownBatches = segmentedBrownOnly && !segmentedFlatStop;
    const long segmentBoundary = (owner + 1 < ctx.size) ? segmentEnd : ctx.n;
    const long segmentStop =
        useBrownBatches
            ? std::min<long>(ctx.n, segmentBoundary + brownCols)
            : ((owner + 1 < ctx.size)
                   ? std::min<long>(
                         ctx.n,
                         segmentEnd + ((segmentedFlatStop && !flatStopUsesBrown) ? 0 : brownCols))
                   : ctx.n);
    const long sweepStart = 0;
    const long sweepEndLimit =
        useBrownBatches
            ? std::max<long>(0, segmentEnd - 1)
            : ((segmentedFlatStop && owner + 1 < ctx.size)
            ? std::max<long>(0, segmentEnd - 2 * ctx.b)
            : std::max<long>(0, segmentEnd - 1));
    const long sweepEnd = std::min<long>(ctx.n - 2, sweepEndLimit);

    if (ctx.rank == 0) {
      std::cout << "BC segmented owner " << owner
                << " segment=[" << segmentStart << "," << segmentEnd << ")"
                << " stop=" << segmentStop
                << " sweeps=[" << sweepStart << "," << sweepEnd << ")"
                << std::endl;
    }
    if (ctx.rank == owner && sweepEnd > sweepStart) {
      std::cout << "BC segmented launch owner " << owner << std::endl;
      if (debugChangedColumns) {
        EVD_CUDA_CHECK(cudaMemcpy(dBandBeforeOwner,
                         dBand,
                         sizeof(double) * static_cast<size_t>(2 * ctx.b) *
                             static_cast<size_t>(ctx.n),
                         cudaMemcpyDeviceToDevice));
      }
      int zero = 0;
      EVD_CUDA_CHECK(cudaMemcpyToSymbol(bcStopFlag, &zero, sizeof(int)));
	      EVD_CUDA_CHECK(cudaMemcpy(com, hCom.data(), sizeof(int) * hCom.size(), cudaMemcpyHostToDevice));
	      const long activeSweeps = sweepEnd - sweepStart;
	      const bool ownerSerialAllowsBatching = segmentedFlatStop || useBrownBatches;
	      const long batchSweeps =
	          (ownerSerialAllowsBatching && requestedBatchSweeps > 0)
	              ? std::min(requestedBatchSweeps, activeSweeps)
	              : activeSweeps;
      if (batchSweeps <= 1 && activeSweeps > 1) {
        if (ctx.rank == 0) {
          std::cerr << "EVD_BC_SEGMENTED_BATCH_SWEEPS must be at least 2 when batching is enabled"
                    << std::endl;
        }
        MPI_Abort(ctx.comm, 2);
      }
      const long brownBatchSweeps =
          useBrownBatches
              ? std::min<long>((requestedBatchSweeps > 0)
                                   ? requestedBatchSweeps
                                   : std::max<long>(1, brownCapacitySweeps),
                               activeSweeps)
              : batchSweeps;
      long maxSegmentRetries =
          static_cast<long>(envIntOrDefault("EVD_BC_SEGMENTED_MAX_RETRIES", 4096));
      if (maxSegmentRetries < 1) {
        maxSegmentRetries = 1;
      }
      for (long batchEnd = std::min(sweepEnd, sweepStart + brownBatchSweeps);
           batchEnd <= sweepEnd;
           batchEnd = std::min(sweepEnd, batchEnd + brownBatchSweeps)) {
        int sweepStartInt = static_cast<int>(sweepStart);
        int sweepEndInt = static_cast<int>(batchEnd);
        int segmentStartInt = segmentedFlatStop ? ((owner + 1 < ctx.size) ? -1 : -2) : 0;
        int segmentEndInt = static_cast<int>(segmentStop);
        int segmentStopSweepInt = -1;
        if (printSegmentedBatch && ctx.rank == owner) {
          std::cout << "BC segmented batch owner " << owner
                    << " sweeps=[" << sweepStartInt << "," << sweepEndInt << ")"
                    << " segment_end=" << segmentEndInt
                    << std::endl;
        }
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
        dim3 dimBlock(32, 32, 1);
        dim3 dimGrid(blockNum, 1, 1);
        long retry = 0;
        bool segmentReached = false;
        long long prevProgressScore = -1;
        do {
          int zeroBatch = 0;
          EVD_CUDA_CHECK(cudaMemcpyToSymbol(bcStopFlag, &zeroBatch, sizeof(int)));
          EVD_CUDA_CHECK(cudaLaunchCooperativeKernel((void *)chaseBulgesSegmentPackedKernel<kBcBandwidth>,
                                            dimGrid,
                                            dimBlock,
                                            kernelArgs));
          EVD_CUDA_CHECK(cudaGetLastError());
          EVD_CUDA_CHECK(cudaDeviceSynchronize());
          if (!segmentedFlatStop) {
            break;
          }
          EVD_CUDA_CHECK(cudaMemcpy(hCom.data(), com, sizeof(int) * hCom.size(), cudaMemcpyDeviceToHost));
          long reached = 0;
          long long progressScore = 0;
          for (long sweep = sweepStart; sweep < batchEnd; ++sweep) {
            const long progress = hCom[static_cast<size_t>(sweep)];
            if (progress >= segmentStop || progress >= ctx.n + 3 * ctx.b) {
              reached++;
            }
            progressScore += std::min<long>(progress, segmentStop);
          }
          segmentReached = (reached == (batchEnd - sweepStart));
          if (segmentReached || progressScore == prevProgressScore) {
            break;
          }
          prevProgressScore = progressScore;
          retry++;
        } while (retry < maxSegmentRetries);
        if (segmentedFlatStop && !segmentReached && ctx.rank == owner) {
          std::cout << "BC segmented flat-stop batch owner " << owner
                    << " sweeps=[" << sweepStartInt << "," << sweepEndInt << ")"
                    << " stopped before segment_end=" << segmentEndInt
                    << " retries=" << (retry + 1)
                    << std::endl;
        }
        if (batchEnd >= sweepEnd) {
          break;
        }
      }
      EVD_CUDA_CHECK(cudaMemcpy(hCom.data(), com, sizeof(int) * hCom.size(), cudaMemcpyDeviceToHost));
      if (debugChangedColumns) {
        const long ldBand = 2 * ctx.b;
        const size_t bandElems = static_cast<size_t>(ldBand) * static_cast<size_t>(ctx.n);
        std::vector<double> before(bandElems);
        std::vector<double> after(bandElems);
        EVD_CUDA_CHECK(cudaMemcpy(before.data(),
                         dBandBeforeOwner,
                         sizeof(double) * bandElems,
                         cudaMemcpyDeviceToHost));
        EVD_CUDA_CHECK(cudaMemcpy(after.data(),
                         dBand,
                         sizeof(double) * bandElems,
                         cudaMemcpyDeviceToHost));
        long firstChangedCol = ctx.n;
        long lastChangedCol = -1;
        long maxDiffCol = -1;
        long maxDiffRow = -1;
        double maxDiff = 0.0;
        for (long col = 0; col < ctx.n; ++col) {
          bool colChanged = false;
          for (long row = 0; row < ldBand; ++row) {
            const size_t idx = static_cast<size_t>(col) * static_cast<size_t>(ldBand) +
                               static_cast<size_t>(row);
            const double diff = std::abs(after[idx] - before[idx]);
            if (diff > debugChangedColumnsEps) {
              colChanged = true;
            }
            if (diff > maxDiff) {
              maxDiff = diff;
              maxDiffCol = col;
              maxDiffRow = row;
            }
          }
          if (colChanged) {
            firstChangedCol = std::min(firstChangedCol, col);
            lastChangedCol = std::max(lastChangedCol, col);
          }
        }
        if (lastChangedCol >= firstChangedCol) {
          std::cout << "BC segmented changed columns owner " << owner
                    << " first=" << firstChangedCol
                    << " last=" << lastChangedCol
                    << " count=" << (lastChangedCol - firstChangedCol + 1)
                    << " max_diff=" << std::setprecision(17) << maxDiff
                    << " max_col=" << maxDiffCol
                    << " max_row_offset=" << maxDiffRow
                    << " eps=" << debugChangedColumnsEps
                    << std::endl;
        } else {
          std::cout << "BC segmented changed columns owner " << owner
                    << " none eps=" << debugChangedColumnsEps << std::endl;
        }
      }
      std::cout << "BC segmented done owner " << owner << std::endl;
    }

    if (segmentedOwnerSerialPackedUOwnerBcast) {
      comBroadcastInts += static_cast<unsigned long long>(hCom.size());
      MPI_CHECK(MPI_Bcast(hCom.data(), static_cast<int>(hCom.size()), MPI_INT, owner, ctx.comm));
    } else if (segmentedComChain) {
      handoffComProgressToNextOwner(ctx, &hCom, owner);
    } else {
      MPI_CHECK(MPI_Bcast(hCom.data(), static_cast<int>(hCom.size()), MPI_INT, owner, ctx.comm));
    }

    if (ctx.rank == owner) {
      long progressMin = ctx.n + 3 * ctx.b;
      long progressMax = 0;
      long completed = 0;
      long reachedStop = 0;
      for (long sweep = sweepStart; sweep < sweepEnd; ++sweep) {
        const long progress = hCom[static_cast<size_t>(sweep)];
        progressMin = std::min(progressMin, progress);
        progressMax = std::max(progressMax, progress);
        if (progress >= ctx.n + 3 * ctx.b) {
          completed++;
        }
        if (progress >= segmentStop) {
          reachedStop++;
        }
      }
	      std::cout << "BC segmented progress owner " << owner
	                << " active_sweeps=" << (sweepEnd - sweepStart)
	                << " batch_sweeps="
	                << (useBrownBatches
	                        ? std::min<long>((requestedBatchSweeps > 0)
	                                             ? requestedBatchSweeps
	                                             : std::max<long>(1, brownCapacitySweeps),
	                                         std::max<long>(1, sweepEnd - sweepStart))
	                        : ((segmentedFlatStop && requestedBatchSweeps > 0)
	                        ? std::min<long>(requestedBatchSweeps, std::max<long>(1, sweepEnd - sweepStart))
	                        : (sweepEnd - sweepStart)))
                << " completed=" << completed
                << " reached_stop=" << reachedStop
                << " progress_min=" << progressMin
                << " progress_max=" << progressMax
                << std::endl;
    }

    if (segmentedFullBandSync) {
      bandBroadcastDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                              static_cast<unsigned long long>(ctx.n);
      broadcastBandSuffix(ctx, dBand, 0, owner);
    } else if (owner + 1 < ctx.size) {
      if (!segmentedBrownOnly && !segmentedFlatStop) {
        const long bandStart = 0;
        const long bandEnd = ctx.n;
        const long bandCols = std::max<long>(0, bandEnd - bandStart);
        maxBandNeighborCols = std::max(maxBandNeighborCols, bandCols);
        if (bandCols >= ctx.n) {
          fullBandNeighborWindows++;
        }
        bandNeighborDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                               static_cast<unsigned long long>(bandCols);
        handoffBandRangeToNextOwner(ctx, dBand, bandStart, bandEnd, owner);
      } else if (segmentedFlatStop) {
        const long bandStart = 0;
        const long bandEnd = std::min<long>(ctx.n, segmentStop + 2 * ctx.b);
        const long bandCols = std::max<long>(0, bandEnd - bandStart);
        maxBandNeighborCols = std::max(maxBandNeighborCols, bandCols);
        if (bandCols >= ctx.n) {
          fullBandNeighborWindows++;
        }
        if (bandEnd > bandStart) {
          bandNeighborDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                                  static_cast<unsigned long long>(bandCols);
        }
        handoffBandRangeToNextOwner(ctx, dBand, bandStart, bandEnd, owner);
      } else {
        const long ownerBrownBatchSweeps =
            useBrownBatches
                ? std::min<long>((requestedBatchSweeps > 0)
                                     ? requestedBatchSweeps
                                     : std::max<long>(1, brownCapacitySweeps),
                                 std::max<long>(1, sweepEnd - sweepStart))
                : 0;
        const long slantRows = useBrownBatches
                                   ? 2 * ctx.b * std::max<long>(0, ownerBrownBatchSweeps - 1)
                                   : 2 * ctx.b * std::max<long>(0, std::max<long>(0, sweepEnd - sweepStart) - 1);
        const long bandStart = std::max<long>(0, segmentEnd - ctx.b - slantRows);
        const long bandEnd = std::min<long>(ctx.n, segmentStop + ctx.b + slantRows);
        const long bandCols = std::max<long>(0, bandEnd - bandStart);
        maxBandNeighborCols = std::max(maxBandNeighborCols, bandCols);
        if (bandCols >= ctx.n) {
          fullBandNeighborWindows++;
        }
        if (bandEnd > bandStart) {
          bandNeighborDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                                  static_cast<unsigned long long>(bandCols);
        }
        handoffBandRangeToNextOwner(ctx, dBand, bandStart, bandEnd, owner);
      }
    }
		    if (segmentedOwnerSerialPackedUOwnerBcast) {
		      PackedUProgressPlan progressPlan =
		          buildPackedUProgressPlan(ctx, hComBeforeOwner, hCom);
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
	    } else if (segmentedPackedUChain && owner + 1 < ctx.size) {
	      if (segmentedPackedUCompactChain) {
	        packedUNeighborDoubles +=
	            static_cast<unsigned long long>(cumulativePackedUPlans[owner].totalElems);
        handoffPackedUSlicesToNextOwnerCompact(ctx,
                                               dUPacked,
                                               dPackedUCompactChain,
                                               cumulativePackedUPlans[owner],
                                               owner);
      } else {
        packedUNeighborDoubles += static_cast<unsigned long long>(packedUElems);
	        handoffDeviceDoublesToNextOwner(ctx, dUPacked, packedUElems, owner);
	      }
	    }
	  }
	  if (segmentedOwnerGatherBack && !segmentedFullBandSync &&
	      !segmentedBrownOnly && !segmentedFlatStop && ctx.size > 1) {
	    const int finalOwner = ctx.size - 1;
	    for (int target = 0; target < finalOwner; ++target) {
	      const long ownedStart = ctx.displs[target];
	      const long ownedEnd = ctx.displs[target] + ctx.counts[target];
	      const long cols = std::max<long>(0, ownedEnd - ownedStart);
	      if (cols <= 0) {
	        continue;
	      }
	      maxBandNeighborCols = std::max(maxBandNeighborCols, cols);
	      bandOwnerHandoffDoubles += static_cast<unsigned long long>(2 * ctx.b) *
	                                 static_cast<unsigned long long>(cols);
	      handoffBandRangeToPeer(ctx, dBand, ownedStart, ownedEnd, finalOwner, target);
	    }
	  }
	  }

	  if (segmentedTileWave && segmentedTileOwnerHandoff &&
      envIntOrDefault("EVD_STAGE_PIPELINE", 0) != 0) {
    MPI_CHECK(MPI_Barrier(ctx.comm));
  }

