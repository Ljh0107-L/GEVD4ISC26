	  } else if (segmentedRolling) {
    if (debugStageProgress) {
      std::cout << "StageProgress rank=" << ctx.rank
                << " phase=bc_rolling_begin local_cols=" << ctx.localCols
                << " col_start=" << ctx.colStart << std::endl;
    }
    long rollingBatchSweeps =
        (requestedBatchSweeps > 0) ? requestedBatchSweeps : std::max<long>(1, brownCapacitySweeps);
    if (rollingBatchSweeps < 1) {
      rollingBatchSweeps = 1;
    }
    const long totalSweeps = std::max<long>(0, ctx.n - 2);
    std::vector<long> segmentSweepEnds(ctx.size, 0);
    for (int owner = 0; owner < ctx.size; ++owner) {
      const long segmentEnd = ctx.displs[owner] + ctx.counts[owner];
      segmentSweepEnds[owner] = std::min<long>(totalSweeps, std::max<long>(0, segmentEnd - 1));
    }
    if (ctx.rank == 0) {
      std::cout << "BC segmented rolling batch_sweeps=" << rollingBatchSweeps
                << " total_sweeps=" << totalSweeps
                << " flat_stop=" << (segmentedRollingFlatStop ? 1 : 0)
                << " block_stop=" << (segmentedRollingBlockStop ? 1 : 0)
                << " frontier_window=" << (segmentedRollingFrontierWindow ? 1 : 0)
                << " dirty_handoff=" << (segmentedRollingDirtyHandoff ? 1 : 0)
                << " dirty_bcast=" << (segmentedRollingDirtyBcast ? 1 : 0)
                << " dirty_prefix=" << (segmentedRollingDirtyPrefix ? 1 : 0)
                << " pull_handoff=" << (segmentedRollingPullHandoff ? 1 : 0)
                << " dirty_chain_all=" << (segmentedRollingDirtyChainAll ? 1 : 0)
                << " dirty_chain_mode=" << segmentedRollingDirtyChainMode
                << std::endl;
    }

    if (!segmentedComChain && ctx.rank == 0) {
      std::cerr << "EVD_BC_SEGMENTED_ROLLING=1 requires EVD_BC_SEGMENTED_COM_CHAIN=1 "
                << "because rolling wavefront must preserve each owner's local previous-batch "
                << "progress while receiving only the current batch from the previous owner"
                << std::endl;
    }
    if (!segmentedComChain) {
      MPI_Abort(ctx.comm, 2);
    }
    if (!segmentedRollingBackSync && ctx.rank == 0) {
      std::cerr << "EVD_BC_SEGMENTED_ROLLING=1 requires "
                << "EVD_BC_SEGMENTED_ROLLING_BACK_SYNC=1; diagonal rolling uses the "
                << "previous owner's next batch only after the current owner returns the "
                << "updated brown/slant frontier" << std::endl;
    }
    if (!segmentedRollingBackSync) {
      MPI_Abort(ctx.comm, 2);
    }
    long maxSegmentRetries =
        static_cast<long>(envIntOrDefault("EVD_BC_SEGMENTED_MAX_RETRIES", 4096));
    if (maxSegmentRetries < 1) {
      maxSegmentRetries = 1;
    }

	    const long rollingBatchCount = ceilDiv(totalSweeps, rollingBatchSweeps);
	    long rollingChangedColumnPrints = 0;
	    std::vector<int> rollingBandLatestOwners;
	    if (segmentedRollingPullHandoff) {
	      const bool stagePipelineForSbrInput =
	          envIntOrDefault("EVD_STAGE_PIPELINE", 0) != 0;
	      const bool sbrBandPipelineInput =
	          envIntOrDefault("EVD_SBR_BAND_PIPELINE",
	                          stagePipelineForSbrInput ? 1 : 0) != 0;
	      const bool sbrPipelineEarlyReturn =
	          envIntOrDefault("EVD_SBR_PIPELINE_EARLY_RETURN",
	                          stagePipelineForSbrInput ? 1 : 0) != 0;
	      const long sbrPipelineGuardCols =
	          sbrPipelineEarlyReturn ? computeSbrPipelineGuardCols(ctx.n, ctx.b, ctx.nb) : 0;
	      rollingBandLatestOwners.resize(static_cast<size_t>(ctx.n));
	      for (long col = 0; col < ctx.n; ++col) {
	        int sourceOwner = ownerOfColumn(col, ctx.counts, ctx.displs);
	        if (sbrBandPipelineInput) {
	          const long blockStart = (ctx.nb > 0) ? (col / ctx.nb) * ctx.nb : col;
	          const long guardedBlockStart =
	              sbrPipelineEarlyReturn ? std::max<long>(0, blockStart - sbrPipelineGuardCols) : 0;
	          sourceOwner =
	              encodeSuffixBandAvailability(ownerOfColumn(guardedBlockStart,
	                                                         ctx.counts,
	                                                         ctx.displs));
	        } else {
	          sourceOwner = encodeSuffixBandAvailability(0);
	        }
	        rollingBandLatestOwners[static_cast<size_t>(col)] = sourceOwner;
	      }
	    }
	    auto rollingDirtyBandWindow = [&](int dirtyOwner,
	                                      long dirtyBatchStart,
	                                      long dirtyBatchSweeps) {
      const long dirtySegmentStart = ctx.displs[dirtyOwner];
      const long dirtySegmentEnd = ctx.displs[dirtyOwner] + ctx.counts[dirtyOwner];
      const long dirtySlantRows = 2 * ctx.b * std::max<long>(0, dirtyBatchSweeps - 1);
      const long dirtyStartBase =
          (dirtyOwner == 0) ? 0 : dirtySegmentStart - 1;
      const long dirtyEndBase = (dirtyOwner + 1 < ctx.size)
                                    ? dirtySegmentEnd +
                                          (segmentedRollingBlockStop ? dirtySlantRows
                                                                     : (brownCols + dirtySlantRows))
                                    : ctx.n;
      long dirtyStart = dirtyStartBase;
      long dirtyEnd = dirtyEndBase + dirtyBatchStart + 1;
      if (segmentedRollingDirtyPrefix) {
        dirtyStart = 0;
      }
      dirtyStart = std::max<long>(0, std::min<long>(ctx.n, dirtyStart));
      dirtyEnd = std::max<long>(dirtyStart, std::min<long>(ctx.n, dirtyEnd));
	      return std::pair<long, long>(dirtyStart, dirtyEnd);
	    };
    auto rollingFrontierBandWindow = [&](int dirtyOwner,
                                         long dirtyBatchSweeps,
                                         bool toNextOwner) {
      if ((toNextOwner && dirtyOwner + 1 >= ctx.size) ||
          (!toNextOwner && dirtyOwner <= 0)) {
        return std::pair<long, long>(0, 0);
      }
      const long dirtySegmentStart = ctx.displs[dirtyOwner];
      const long dirtySegmentEnd = ctx.displs[dirtyOwner] + ctx.counts[dirtyOwner];
      const long boundary = toNextOwner ? dirtySegmentEnd : dirtySegmentStart;
      const long dirtySlantRows = 2 * ctx.b * std::max<long>(0, dirtyBatchSweeps - 1);
      const long rightHalo = ctx.b + dirtySlantRows +
                             (segmentedRollingBlockStop ? 0 : brownCols);
      const long leftHalo = ctx.b + dirtySlantRows;
      const long dirtyStart =
          std::max<long>(0, std::min<long>(ctx.n, boundary - leftHalo));
      const long dirtyEnd =
          std::max<long>(dirtyStart, std::min<long>(ctx.n, boundary + rightHalo));
      return std::pair<long, long>(dirtyStart, dirtyEnd);
    };
    for (long wave = 0; wave < rollingBatchCount + ctx.size - 1; ++wave) {
      const int ownerBegin = static_cast<int>(std::max<long>(0, wave - (rollingBatchCount - 1)));
      const int ownerEnd = static_cast<int>(std::min<long>(ctx.size - 1, wave));
      for (int owner = ownerEnd; owner >= ownerBegin; --owner) {
        const long batchIndex = wave - owner;
        if (batchIndex < 0 || batchIndex >= rollingBatchCount) {
          continue;
        }
        const long batchStart = batchIndex * rollingBatchSweeps;
        const long batchEnd = std::min<long>(totalSweeps, batchStart + rollingBatchSweeps);
        const long ownerSweepEnd = std::min<long>(batchEnd, segmentSweepEnds[owner]);
        if (ownerSweepEnd <= batchStart) {
          continue;
        }

        const long segmentStart = ctx.displs[owner];
        const long segmentEnd = ctx.displs[owner] + ctx.counts[owner];
        const long segmentBoundary = (owner + 1 < ctx.size) ? segmentEnd : ctx.n;
        const long segmentStop =
            (owner + 1 < ctx.size)
                ? std::min<long>(ctx.n,
                                 segmentBoundary +
                                     (segmentedRollingBlockStop ? 0 : brownCols))
                : ctx.n;

        if (printSegmentedBatch && ctx.rank == 0) {
          std::cout << "BC segmented rolling wave " << wave
                    << " owner " << owner
                    << " batch=[" << batchStart << "," << ownerSweepEnd << ")"
                    << " segment=[" << segmentStart << "," << segmentEnd << ")"
                    << " stop=" << segmentStop << std::endl;
        }
	        if (debugStageProgress) {
	          std::cout << "StageProgress rank=" << ctx.rank
                    << " phase=bc_rolling_owner_enter wave=" << wave
                    << " owner=" << owner
                    << " batch=[" << batchStart << "," << ownerSweepEnd << ")"
                    << " segment=[" << segmentStart << "," << segmentEnd << ")"
                    << " stop=" << segmentStop
	                    << std::endl;
	        }

	        const long ownerBatchSweeps = ownerSweepEnd - batchStart;
	        long ownerDirtyBandStart = 0;
	        long ownerDirtyBandEnd = ctx.n;
	        if (segmentedRollingDirtyHandoff || segmentedRollingDirtyBcast ||
	            segmentedRollingPullHandoff) {
	          const auto dirtyWindow =
	              rollingDirtyBandWindow(owner, batchStart, ownerBatchSweeps);
	          ownerDirtyBandStart = dirtyWindow.first;
	          ownerDirtyBandEnd = dirtyWindow.second;
	        }
	        if (segmentedRollingPullHandoff) {
	          bandPullHandoffDoubles +=
	              pullBandRangeToOwner(ctx,
	                                   dBand,
	                                   ownerDirtyBandStart,
		                                   ownerDirtyBandEnd,
		                                   owner,
		                                   rollingBandLatestOwners,
		                                   &maxBandNeighborCols,
		                                   &bandPullCallCount,
		                                   &bandPullTransferCount,
		                                   &bandPullMaxTransfersPerCall);
	        }
	
	        if (ctx.rank == owner) {
	          if (debugStageProgress) {
            std::cout << "StageProgress rank=" << ctx.rank
                      << " phase=bc_rolling_kernel_begin wave=" << wave
                      << " owner=" << owner
                      << " batch=[" << batchStart << "," << ownerSweepEnd << ")"
                      << std::endl;
          }
          const bool debugThisLaunch =
              debugChangedColumns && rollingChangedColumnPrints < debugChangedColumnsMaxPrints;
          if (debugThisLaunch) {
            EVD_CUDA_CHECK(cudaMemcpy(dBandBeforeOwner,
                             dBand,
                             sizeof(double) * static_cast<size_t>(2 * ctx.b) *
                                 static_cast<size_t>(ctx.n),
                             cudaMemcpyDeviceToDevice));
          }
          int sweepStartInt = static_cast<int>(batchStart);
          int sweepEndInt = static_cast<int>(ownerSweepEnd);
          int segmentStartInt = segmentedRollingFlatStop ? -5 : -4;
          int segmentEndInt = static_cast<int>(segmentStop);
          int segmentStopSweepInt =
              segmentedRollingFlatStop ? -1 : static_cast<int>(ownerSweepEnd - 1);
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
            int zero = 0;
            EVD_CUDA_CHECK(cudaMemcpyToSymbol(bcStopFlag, &zero, sizeof(int)));
            EVD_CUDA_CHECK(cudaMemcpy(com, hCom.data(), sizeof(int) * hCom.size(), cudaMemcpyHostToDevice));
            EVD_CUDA_CHECK(cudaLaunchCooperativeKernel((void *)chaseBulgesSegmentPackedKernel<kBcBandwidth>,
                                              dimGrid,
                                              dimBlock,
                                              kernelArgs));
            EVD_CUDA_CHECK(cudaGetLastError());
            EVD_CUDA_CHECK(cudaDeviceSynchronize());
            EVD_CUDA_CHECK(cudaMemcpy(hCom.data(), com, sizeof(int) * hCom.size(), cudaMemcpyDeviceToHost));
            long reached = 0;
            long long progressScore = 0;
            for (long sweep = batchStart; sweep < ownerSweepEnd; ++sweep) {
              const long sweepLag = std::max<long>(0, ownerSweepEnd - 1 - sweep);
              const long target =
                  segmentedRollingFlatStop
                      ? segmentStop
                      : std::min<long>(ctx.n, segmentStop + 2 * ctx.b * sweepLag);
              const long progress = hCom[static_cast<size_t>(sweep)];
              if (progress >= target || progress >= ctx.n + 3 * ctx.b) {
                reached++;
              }
              progressScore += std::min<long>(progress, target);
            }
            segmentReached = (reached == (ownerSweepEnd - batchStart));
            if (segmentReached || progressScore == prevProgressScore) {
              break;
            }
            prevProgressScore = progressScore;
            retry++;
          } while (retry < maxSegmentRetries);
          if (!segmentReached && printSegmentedBatch) {
            std::cout << "BC segmented rolling step owner " << owner
                      << " batch=[" << batchStart << "," << ownerSweepEnd << ")"
                      << " stopped before segment_stop=" << segmentStop
                      << " retries=" << (retry + 1)
                      << std::endl;
          }
          if (debugStageProgress) {
            std::cout << "StageProgress rank=" << ctx.rank
                      << " phase=bc_rolling_kernel_done wave=" << wave
                      << " owner=" << owner
                      << " segment_reached=" << (segmentReached ? 1 : 0)
                      << " retries=" << (retry + 1)
                      << std::endl;
          }
          if (debugThisLaunch) {
            std::string label =
                "BC segmented rolling changed columns wave=" + std::to_string(wave) +
                " owner=" + std::to_string(owner) +
                " batch=[" + std::to_string(batchStart) + "," +
                std::to_string(ownerSweepEnd) + ")";
            printChangedBandColumns(label,
                                    ctx.n,
                                    ctx.b,
                                    dBandBeforeOwner,
                                    dBand,
                                    debugChangedColumnsEps);
            rollingChangedColumnPrints++;
	          }
	        }
	
	        if (segmentedRollingFullBcast) {
          if (debugStageProgress) {
            std::cout << "StageProgress rank=" << ctx.rank
                      << " phase=bc_rolling_full_band_bcast owner=" << owner
                      << std::endl;
          }
          bandBroadcastDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                                  static_cast<unsigned long long>(ctx.n);
          broadcastBandSuffix(ctx, dBand, 0, owner);
        } else if (segmentedRollingDirtyBcast) {
          const long dirtyCols = std::max<long>(0, ownerDirtyBandEnd - ownerDirtyBandStart);
          maxBandNeighborCols = std::max(maxBandNeighborCols, dirtyCols);
          if (dirtyCols > 0) {
            bandBroadcastDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                                    static_cast<unsigned long long>(dirtyCols);
          }
          broadcastBandColumnRange(ctx, dBand, ownerDirtyBandStart, ownerDirtyBandEnd, owner);
        }
	        if (segmentedRollingFullComBcast) {
          if (debugStageProgress) {
            std::cout << "StageProgress rank=" << ctx.rank
                      << " phase=bc_rolling_full_com_bcast owner=" << owner
                      << std::endl;
          }
          comBroadcastInts += static_cast<unsigned long long>(hCom.size());
          MPI_CHECK(MPI_Bcast(hCom.data(),
                              static_cast<int>(hCom.size()),
                              MPI_INT,
                              owner,
                              ctx.comm));
	        }
	
	        if (segmentedRollingDirtyHandoff && segmentedRollingDirtyChainMode != 0 &&
	            !segmentedRollingFullBcast &&
	            !segmentedRollingDirtyBcast) {
	          const bool chainPrefix = !segmentedRollingPullHandoff &&
	                                   (segmentedRollingDirtyChainMode == 1 ||
	                                    segmentedRollingDirtyChainMode == 3 ||
	                                    segmentedRollingDirtyChainMode == 5);
	          const bool chainSuffix = !segmentedRollingPullHandoff &&
	                                   (segmentedRollingDirtyChainMode == 1 ||
	                                    segmentedRollingDirtyChainMode == 2 ||
	                                    segmentedRollingDirtyChainMode == 4);
	          const bool immediatePrefix = segmentedRollingDirtyChainMode == 4;
	          const bool immediateSuffix = !segmentedRollingPullHandoff &&
	                                       segmentedRollingDirtyChainMode == 5;
          const auto prevDirtyWindow =
              segmentedRollingFrontierWindow
                  ? rollingFrontierBandWindow(owner, ownerBatchSweeps, false)
                  : std::pair<long, long>(ownerDirtyBandStart, ownerDirtyBandEnd);
          const auto nextDirtyWindow =
              segmentedRollingFrontierWindow
                  ? rollingFrontierBandWindow(owner, ownerBatchSweeps, true)
                  : std::pair<long, long>(ownerDirtyBandStart, ownerDirtyBandEnd);
          auto countDirtyWindowDoubles = [&](const std::pair<long, long> &window) {
            const long cols = std::max<long>(0, window.second - window.first);
            maxBandNeighborCols = std::max(maxBandNeighborCols, cols);
            return static_cast<unsigned long long>(2 * ctx.b) *
                   static_cast<unsigned long long>(cols);
          };
          if (debugStageProgress) {
            std::cout << "StageProgress rank=" << ctx.rank
                      << " phase=bc_rolling_dirty_handoff owner=" << owner
                      << " dirty_cols=[" << ownerDirtyBandStart << "," << ownerDirtyBandEnd << ")"
                      << " prev_frontier=[" << prevDirtyWindow.first << ","
                      << prevDirtyWindow.second << ")"
                      << " next_frontier=[" << nextDirtyWindow.first << ","
                      << nextDirtyWindow.second << ")"
                      << " frontier_window=" << (segmentedRollingFrontierWindow ? 1 : 0)
                      << " chain_mode=" << segmentedRollingDirtyChainMode
                      << std::endl;
          }
          if (chainPrefix) {
            const unsigned long long dirtyDoubles =
                countDirtyWindowDoubles(prevDirtyWindow);
            if (dirtyDoubles > 0) {
              bandNeighborBackDoubles +=
                  dirtyDoubles * static_cast<unsigned long long>(owner);
              handoffBandRangeToOwnerPrefixByNeighborChain(ctx,
                                                           dBand,
                                                           prevDirtyWindow.first,
                                                           prevDirtyWindow.second,
                                                           owner);
            }
          } else if (immediatePrefix && owner > 0) {
            const unsigned long long dirtyDoubles =
                countDirtyWindowDoubles(prevDirtyWindow);
            if (dirtyDoubles > 0) {
              bandNeighborBackDoubles += dirtyDoubles;
              handoffBandRangeToPrevOwner(ctx,
                                          dBand,
                                          prevDirtyWindow.first,
                                          prevDirtyWindow.second,
                                          owner);
            }
          }
          if (chainSuffix) {
            const unsigned long long dirtyDoubles =
                countDirtyWindowDoubles(nextDirtyWindow);
            if (dirtyDoubles > 0) {
              bandNeighborDoubles +=
                  dirtyDoubles * static_cast<unsigned long long>(ctx.size - owner - 1);
              handoffBandRangeToOwnerSuffixByNeighborChain(ctx,
                                                           dBand,
                                                           nextDirtyWindow.first,
                                                           nextDirtyWindow.second,
                                                           owner);
            }
          } else if (immediateSuffix && owner + 1 < ctx.size) {
            const unsigned long long dirtyDoubles =
                countDirtyWindowDoubles(nextDirtyWindow);
            if (dirtyDoubles > 0) {
              bandNeighborDoubles += dirtyDoubles;
              handoffBandRangeToNextOwner(ctx,
                                          dBand,
                                          nextDirtyWindow.first,
                                          nextDirtyWindow.second,
                                          owner);
	            }
	          }
	        }
	        if (segmentedRollingPullHandoff) {
	          markBandRangeOwner(ctx.n,
	                             ownerDirtyBandStart,
	                             ownerDirtyBandEnd,
	                             owner,
	                             &rollingBandLatestOwners);
	        }
	
	        if (owner > 0) {
          const int prevOwner = owner - 1;
          const long prevSegmentEnd = ctx.displs[prevOwner] + ctx.counts[prevOwner];
          const long prevSegmentStop =
              (prevOwner + 1 < ctx.size)
                  ? std::min<long>(ctx.n,
                                   prevSegmentEnd +
                                       (segmentedRollingBlockStop ? 0 : brownCols))
                                         : ctx.n;
          const long prevOwnerSweepEnd = std::min<long>(batchEnd, segmentSweepEnds[prevOwner]);
          const long prevBatchSweeps = std::max<long>(0, prevOwnerSweepEnd - batchStart);
          if (prevBatchSweeps > 0) {
            if (!segmentedRollingFullComBcast) {
              if (debugStageProgress) {
                std::cout << "StageProgress rank=" << ctx.rank
                          << " phase=bc_rolling_prev_com owner=" << owner
                          << " prev_owner=" << prevOwner
                          << " batch=[" << batchStart << "," << prevOwnerSweepEnd << ")"
                          << std::endl;
              }
              handoffComRangeToPrevOwner(ctx, &hCom, batchStart, prevOwnerSweepEnd, owner);
            }

            const long prevSlantRows = 2 * ctx.b * std::max<long>(0, prevBatchSweeps - 1);
            long backBandStart = std::max<long>(0, prevSegmentEnd - ctx.b - prevSlantRows);
            long backBandEnd = std::min<long>(ctx.n, prevSegmentStop + ctx.b + prevSlantRows);
            if (segmentedRollingDirtyHandoff) {
              backBandStart = ownerDirtyBandStart;
              backBandEnd = ownerDirtyBandEnd;
            }
            const long backBandCols = std::max<long>(0, backBandEnd - backBandStart);
            maxBandNeighborCols = std::max(maxBandNeighborCols, backBandCols);
            if ((!segmentedRollingDirtyHandoff || segmentedRollingDirtyChainMode == 0) &&
                !segmentedRollingFullBcast && !segmentedRollingDirtyBcast &&
                backBandCols > 0) {
              bandNeighborBackDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                                         static_cast<unsigned long long>(backBandCols);
            }
            if ((!segmentedRollingDirtyHandoff || segmentedRollingDirtyChainMode == 0) &&
                !segmentedRollingFullBcast && !segmentedRollingDirtyBcast) {
              handoffBandRangeToPrevOwner(ctx, dBand, backBandStart, backBandEnd, owner);
            }
          }
        }

        if (owner + 1 < ctx.size) {
          if (!segmentedRollingFullComBcast) {
            if (debugStageProgress) {
              std::cout << "StageProgress rank=" << ctx.rank
                        << " phase=bc_rolling_next_com owner=" << owner
                        << " next_owner=" << (owner + 1)
                        << " batch=[" << batchStart << "," << ownerSweepEnd << ")"
                        << std::endl;
            }
            handoffComRangeToNextOwner(ctx, &hCom, batchStart, ownerSweepEnd, owner);
          }

          const long batchSweepsForOwner = ownerSweepEnd - batchStart;
          const long slantRows = 2 * ctx.b * std::max<long>(0, batchSweepsForOwner - 1);
          long bandStart = std::max<long>(0, segmentEnd - ctx.b - slantRows);
          long bandEnd = std::min<long>(ctx.n, segmentStop + ctx.b + slantRows);
          if (segmentedRollingDirtyHandoff) {
            bandStart = ownerDirtyBandStart;
            bandEnd = ownerDirtyBandEnd;
          }
          const long bandCols = std::max<long>(0, bandEnd - bandStart);
          maxBandNeighborCols = std::max(maxBandNeighborCols, bandCols);
          if ((!segmentedRollingDirtyHandoff || segmentedRollingDirtyChainMode == 0) &&
              !segmentedRollingFullBcast && bandCols >= ctx.n) {
            fullBandNeighborWindows++;
          }
          if ((!segmentedRollingDirtyHandoff || segmentedRollingDirtyChainMode == 0) &&
              !segmentedRollingFullBcast && !segmentedRollingDirtyBcast &&
              bandEnd > bandStart) {
            bandNeighborDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                                    static_cast<unsigned long long>(bandCols);
          }
          if ((!segmentedRollingDirtyHandoff || segmentedRollingDirtyChainMode == 0) &&
              !segmentedRollingFullBcast && !segmentedRollingDirtyBcast) {
            handoffBandRangeToNextOwner(ctx, dBand, bandStart, bandEnd, owner);
          }

          if (segmentedPackedUChain) {
            if (segmentedPackedUCompactChain) {
              if (debugStageProgress) {
                std::cout << "StageProgress rank=" << ctx.rank
                          << " phase=bc_rolling_packedu_next owner=" << owner
                          << " next_owner=" << (owner + 1)
                          << " batch=[" << batchStart << "," << ownerSweepEnd << ")"
                          << std::endl;
              }
              PackedUSlicePlan batchPackedUPlan =
                  buildPackedUOwnerSlicePlan(ctx, hPackedUOffsets, batchStart, ownerSweepEnd);
              packedUNeighborDoubles +=
                  static_cast<unsigned long long>(batchPackedUPlan.totalElems);
              handoffPackedUSlicesToNextOwnerCompact(ctx,
                                                     dUPacked,
                                                     dPackedUCompactChain,
                                                     batchPackedUPlan,
                                                     owner);
            } else {
              packedUNeighborDoubles += static_cast<unsigned long long>(packedUElems);
              handoffDeviceDoublesToNextOwner(ctx, dUPacked, packedUElems, owner);
            }
          }
        }
      }
    }
