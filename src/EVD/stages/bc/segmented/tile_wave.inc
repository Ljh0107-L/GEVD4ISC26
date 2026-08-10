  } else if (segmentedTileWave) {
    long tileBatchSweeps =
        (requestedBatchSweeps > 0) ? requestedBatchSweeps : std::max<long>(1, brownCapacitySweeps);
    if (tileBatchSweeps < 1) {
      tileBatchSweeps = 1;
    }
    long tileRows =
        static_cast<long>(envIntOrDefault("EVD_BC_SEGMENTED_TILE_ROWS",
                                          static_cast<int>(brownCols)));
    if (tileRows <= 0) {
      if (ctx.rank == 0) {
        std::cerr << "EVD_BC_SEGMENTED_TILE_ROWS must be positive; got "
                  << tileRows << std::endl;
      }
      MPI_Abort(ctx.comm, 2);
    }
    tileRows = ((tileRows + ctx.b - 1) / ctx.b) * ctx.b;
    const long totalSweeps = std::max<long>(0, ctx.n - 2);
    const long batchCount = ceilDiv(totalSweeps, tileBatchSweeps);
    const long tileCount = ceilDiv(ctx.n, tileRows);
    const bool useTilePullHandoff =
        segmentedTileRangeHandoff && !segmentedTileSweepOwner && segmentedTilePullHandoff;
    const bool segmentedTilePreownerBand =
        envIntOrDefault("EVD_BC_SEGMENTED_TILE_PREOWNER_BAND", 0) != 0;
    const bool segmentedTilePeerComHandoff =
        envIntOrDefault("EVD_BC_SEGMENTED_TILE_PEER_COM_HANDOFF",
                        useTilePullHandoff ? 1 : 0) != 0;
    std::vector<int> tileBandLatestOwners;
    if (useTilePullHandoff) {
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
      tileBandLatestOwners.resize(static_cast<size_t>(ctx.n));
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
        tileBandLatestOwners[static_cast<size_t>(col)] = sourceOwner;
      }
    }
    if (segmentedTileBatchMajor && segmentedTileSweepOwner &&
        segmentedTileRangeBcast && segmentedTileMultiStep &&
        !segmentedTileWaveFullBcast && !segmentedTileOwnerHandoff &&
        tileRows < ctx.n) {
      bool unsafeTileBatch = (2 * tileBatchSweeps > tileRows);
      for (int owner = 1; owner < ctx.size; ++owner) {
        unsafeTileBatch =
            unsafeTileBatch || (ctx.displs[owner] % tileBatchSweeps != 0);
      }
      if (unsafeTileBatch &&
          envIntOrDefault("EVD_ALLOW_UNSAFE_BC_SEGMENTED", 0) == 0) {
        if (ctx.rank == 0) {
          std::cerr << "tile sweep-owner range-bcast requires 2*batch_sweeps <= "
                    << "tile_rows and rank block boundaries aligned to batch_sweeps "
                    << "when tile_rows < n; got batch_sweeps=" << tileBatchSweeps
                    << " tile_rows=" << tileRows
                    << ". Set EVD_ALLOW_UNSAFE_BC_SEGMENTED=1 only for diagnostics."
                    << std::endl;
        }
        MPI_Abort(ctx.comm, 2);
      }
    }
    if (ctx.rank == 0) {
      std::cout << "BC segmented tile-wave batch_sweeps=" << tileBatchSweeps
                << " tile_rows=" << tileRows
                << " total_sweeps=" << totalSweeps
                << " tile_count=" << tileCount
                << " batch_major=" << (segmentedTileBatchMajor ? 1 : 0)
                << " stop_mode=" << segmentedTileStopMode
                << " single_step=" << (segmentedTileSingleStep ? 1 : 0)
                << " multi_step=" << (segmentedTileMultiStep ? 1 : 0)
                << " sweep_owner=" << (segmentedTileSweepOwner ? 1 : 0)
                << " range_bcast=" << (segmentedTileRangeBcast ? 1 : 0)
                << " range_handoff=" << (segmentedTileRangeHandoff ? 1 : 0)
                << " suffix_bcast=" << (segmentedTileSuffixBcast ? 1 : 0)
                << " owner_handoff=" << (segmentedTileOwnerHandoff ? 1 : 0)
                << " owner_handoff_prefix=" << (segmentedTileOwnerHandoffPrefix ? 1 : 0)
                << " sparse_com_sync=" << (segmentedTileSparseComSync ? 1 : 0)
                << " row_owner_packedu_bcast="
                << (segmentedTileRowOwnerPackedUBcast ? 1 : 0)
                << " progress_packedu_bcast="
                << (segmentedTileProgressPackedUBcast ? 1 : 0)
                << " progress_packedu_immediate="
                << (segmentedTileProgressPackedUImmediate ? 1 : 0)
                << " frontier_range=" << (segmentedTileFrontierRange ? 1 : 0)
                << " wrap_gather=" << (segmentedTileWrapGather ? 1 : 0)
                << " pull_handoff=" << (useTilePullHandoff ? 1 : 0)
                << " preowner_band=" << (segmentedTilePreownerBand ? 1 : 0)
                << " peer_com_handoff=" << (segmentedTilePeerComHandoff ? 1 : 0)
                << std::endl;
    }
    long maxSegmentRetries =
        static_cast<long>(envIntOrDefault("EVD_BC_SEGMENTED_MAX_RETRIES", 4096));
    if (maxSegmentRetries < 1) {
      maxSegmentRetries = 1;
    }
    double *dTileProgressPackedUCompact = nullptr;
    long tileProgressPackedUCompactCapacity = 0;
    if (segmentedTileProgressPackedUBcast &&
        !segmentedTileProgressPackedUImmediate) {
      tileProgressPackedUPlans.resize(static_cast<size_t>(ctx.size));
    }
    if (useTilePullHandoff && segmentedTilePreownerBand) {
      for (int target = 0; target < ctx.size; ++target) {
        const long ownedStart = ctx.displs[target];
        const long ownedEnd = ctx.displs[target] + ctx.counts[target];
        bandPreownerHandoffDoubles +=
            pullBandRangeToOwner(ctx,
                                 dBand,
                                 ownedStart,
	                                 ownedEnd,
	                                 target,
	                                 tileBandLatestOwners,
	                                 &maxBandNeighborCols,
	                                 &bandPullCallCount,
	                                 &bandPullTransferCount,
	                                 &bandPullMaxTransfersPerCall);
        markBandRangeOwner(ctx.n, ownedStart, ownedEnd, target, &tileBandLatestOwners);
      }
    }

    const long waveCount =
        segmentedTileBatchMajor ? (batchCount * tileCount) : (batchCount + tileCount - 1);
    for (long wave = 0; wave < waveCount; ++wave) {
      const long tileBegin = segmentedTileBatchMajor
                                 ? (wave % tileCount)
                                 : std::max<long>(0, wave - (batchCount - 1));
      const long tileEndExclusive = segmentedTileBatchMajor
                                        ? tileBegin
                                        : std::min<long>(tileCount - 1, wave);
      for (long tile = tileEndExclusive; tile >= tileBegin; --tile) {
        const long batchIndex = segmentedTileBatchMajor ? (wave / tileCount) : (wave - tile);
        if (batchIndex < 0 || batchIndex >= batchCount) {
          continue;
        }
        const long batchStart = batchIndex * tileBatchSweeps;
        const long batchEnd = std::min<long>(totalSweeps, batchStart + tileBatchSweeps);
        if (batchEnd <= batchStart) {
          continue;
        }
        const long rowTileStart = tile * tileRows;
        const long rowTileEnd = std::min<long>(ctx.n, rowTileStart + tileRows);
        const int owner =
            segmentedTileSweepOwner
                ? ownerOfColumn(std::min<long>(ctx.n - 1, batchStart), ctx.counts, ctx.displs)
                : ownerOfColumn(std::min<long>(ctx.n - 1, rowTileStart), ctx.counts, ctx.displs);
        const long batchSweeps = batchEnd - batchStart;
        const long haloCols = 2 * ctx.b * std::max<long>(0, batchSweeps - 1) + ctx.b;
        const long bandStart = std::max<long>(0, rowTileStart - haloCols);
        const long bandEnd = std::min<long>(ctx.n, rowTileEnd + haloCols);
        const long frontierBandStart =
            segmentedTileFrontierRange
                ? std::max<long>(bandStart, std::max<long>(0, rowTileStart - ctx.b))
                : bandStart;

        if (printSegmentedBatch && ctx.rank == 0) {
          std::cout << "BC segmented tile-wave wave " << wave
                    << " tile " << tile
                    << " owner " << owner
                    << " rows=[" << rowTileStart << "," << rowTileEnd << ")"
                    << " sweeps=[" << batchStart << "," << batchEnd << ")"
                    << std::endl;
        }

        if (useTilePullHandoff) {
          bandPullHandoffDoubles +=
              pullBandRangeToOwner(ctx,
                                   dBand,
                                   frontierBandStart,
	                                   bandEnd,
	                                   owner,
	                                   tileBandLatestOwners,
	                                   &maxBandNeighborCols,
	                                   &bandPullCallCount,
	                                   &bandPullTransferCount,
	                                   &bandPullMaxTransfersPerCall);
        }

        std::vector<int> hComBeforeTile;
        if (segmentedTileProgressPackedUBcast && ctx.rank == owner) {
          hComBeforeTile = hCom;
        }

        if (ctx.rank == owner) {
          int zero = 0;
          EVD_CUDA_CHECK(cudaMemcpyToSymbol(bcStopFlag, &zero, sizeof(int)));
          EVD_CUDA_CHECK(cudaMemcpy(com, hCom.data(), sizeof(int) * hCom.size(), cudaMemcpyHostToDevice));
          int segmentStartInt = segmentedTileSingleStep ? -3 : (segmentedTileMultiStep ? -4 : 0);
          int segmentEndInt =
              segmentedTileSingleStep ? nInt : static_cast<int>(rowTileEnd);
          long segmentStopSweep = batchEnd - 1;
          if (segmentedTileStopMode == 1) {
            segmentStopSweep = batchEnd;
          } else if (segmentedTileStopMode == 2) {
            segmentStopSweep = batchStart;
          }
          int segmentStopSweepInt = static_cast<int>(segmentStopSweep);
          dim3 dimBlock(32, 32, 1);
          dim3 dimGrid(blockNum, 1, 1);
          int sweepStartInt = static_cast<int>(batchStart);
          int sweepEndInt = static_cast<int>(batchEnd);
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
          bool segmentReached = false;
          long long prevProgressScore = -1;
          do {
            int zeroBatch = 0;
            EVD_CUDA_CHECK(cudaMemcpyToSymbol(bcStopFlag, &zeroBatch, sizeof(int)));
            EVD_CUDA_CHECK(cudaMemcpy(com, hCom.data(), sizeof(int) * hCom.size(), cudaMemcpyHostToDevice));
            EVD_CUDA_CHECK(cudaLaunchCooperativeKernel((void *)chaseBulgesSegmentPackedKernel<kBcBandwidth>,
                                              dimGrid,
                                              dimBlock,
                                              kernelArgs));
            EVD_CUDA_CHECK(cudaGetLastError());
            EVD_CUDA_CHECK(cudaDeviceSynchronize());
            EVD_CUDA_CHECK(cudaMemcpy(hCom.data(), com, sizeof(int) * hCom.size(), cudaMemcpyDeviceToHost));
            if (!segmentedTileSingleStep && !segmentedTileMultiStep) {
              segmentReached = true;
              break;
            }
            long reached = 0;
            long long progressScore = 0;
            for (long sweep = batchStart; sweep < batchEnd; ++sweep) {
              const long sweepLag = std::max<long>(0, segmentStopSweep - sweep);
              const long target =
                  std::min<long>(ctx.n, rowTileEnd + 2 * ctx.b * sweepLag);
              const long progress = hCom[static_cast<size_t>(sweep)];
              if (progress >= target || progress >= ctx.n + 3 * ctx.b) {
                reached++;
              }
              progressScore += std::min<long>(progress, target);
            }
            segmentReached = (reached == (batchEnd - batchStart));
            if (segmentReached || progressScore == prevProgressScore) {
              break;
            }
            prevProgressScore = progressScore;
            retry++;
          } while (retry < maxSegmentRetries);
          if ((segmentedTileSingleStep || segmentedTileMultiStep) &&
              !segmentReached && printSegmentedBatch) {
            std::cout << "BC segmented tile-wave single-step owner " << owner
                      << " rows=[" << rowTileStart << "," << rowTileEnd << ")"
                      << " sweeps=[" << batchStart << "," << batchEnd << ")"
                      << " retries=" << (retry + 1)
                      << " stopped before target" << std::endl;
          }
        }

        if (segmentedTileProgressPackedUBcast) {
          PackedUProgressPlan progressPlan;
          if (ctx.rank == owner) {
            progressPlan = buildPackedUProgressPlan(ctx, hComBeforeTile, hCom);
          }
          if (segmentedTileProgressPackedUImmediate) {
            broadcastPackedUProgressPlanMetadata(ctx, &progressPlan, owner);
            if (progressPlan.totalElems > 0) {
              if (progressPlan.totalElems > tileProgressPackedUCompactCapacity) {
                if (dTileProgressPackedUCompact != nullptr) {
                  EVD_CUDA_CHECK(evdFree(dTileProgressPackedUCompact));
                }
                EVD_CUDA_CHECK(evdMalloc(&dTileProgressPackedUCompact,
                                 sizeof(double) *
                                     static_cast<size_t>(progressPlan.totalElems)));
                tileProgressPackedUCompactCapacity = progressPlan.totalElems;
              }
              packedUBroadcastDoubles +=
                  static_cast<unsigned long long>(progressPlan.totalElems);
              recordPackedUProgressPlan(progressPlan);
              broadcastPackedUProgressCompact(ctx,
                                              dUPacked,
                                              dTileProgressPackedUCompact,
                                              progressPlan,
                                              dPackedUOffsets,
                                              packedUSweepCount,
                                              owner,
                                              &packedUProgressWorkspace);
            }
          } else if (ctx.rank == owner && progressPlan.totalElems > 0) {
            appendPackedUProgressPlan(
                &tileProgressPackedUPlans[static_cast<size_t>(owner)],
                progressPlan);
            }
        }

        if (useTilePullHandoff) {
          markBandRangeOwner(ctx.n,
                             frontierBandStart,
                             bandEnd,
                             owner,
                             &tileBandLatestOwners);
        }

        int nextBatchOwner = owner;
        bool ownerHandoffBoundary = false;
        if (segmentedTileOwnerHandoff && tile + 1 == tileCount && batchEnd < totalSweeps) {
          nextBatchOwner =
              ownerOfColumn(std::min<long>(ctx.n - 1, batchEnd), ctx.counts, ctx.displs);
          ownerHandoffBoundary = nextBatchOwner != owner;
        }
        if (segmentedTileOwnerHandoff) {
          if (ownerHandoffBoundary) {
            comOwnerHandoffInts += static_cast<unsigned long long>(hCom.size());
            handoffComProgressToPeer(ctx, &hCom, owner, nextBatchOwner);
          }
        } else if (segmentedTilePeerComHandoff && segmentedTileBatchMajor &&
                   !segmentedTileSweepOwner) {
          int nextComOwner = -1;
          if (tile + 1 < tileCount) {
            nextComOwner =
                ownerOfColumn(std::min<long>(ctx.n - 1, rowTileEnd),
                              ctx.counts,
                              ctx.displs);
          } else if (batchEnd < totalSweeps) {
            nextComOwner = ownerOfColumn(0, ctx.counts, ctx.displs);
          }
          if (nextComOwner >= 0 && nextComOwner != owner) {
            comOwnerHandoffInts += static_cast<unsigned long long>(hCom.size());
            handoffComProgressToPeer(ctx, &hCom, owner, nextComOwner);
          }
        } else {
          bool syncComThisTile = true;
          if (segmentedTileSparseComSync && segmentedTileBatchMajor &&
              segmentedTileSweepOwner) {
            const bool lastTileInBatch = (tile + 1 == tileCount);
            const long nextBatchIndex = batchIndex + 1;
            bool nextBatchNeedsDifferentOwner = false;
            if (lastTileInBatch && nextBatchIndex < batchCount) {
              const long nextBatchStart = nextBatchIndex * tileBatchSweeps;
              const int nextOwner =
                  ownerOfColumn(std::min<long>(ctx.n - 1, nextBatchStart),
                                ctx.counts,
                                ctx.displs);
              nextBatchNeedsDifferentOwner = (nextOwner != owner);
            }
            syncComThisTile = lastTileInBatch && nextBatchNeedsDifferentOwner;
          }
          if (syncComThisTile) {
            comBroadcastInts += static_cast<unsigned long long>(hCom.size());
            MPI_CHECK(MPI_Bcast(hCom.data(),
                                static_cast<int>(hCom.size()),
                                MPI_INT,
                                owner,
                                ctx.comm));
          }
        }
        if (segmentedTileWaveFullBcast) {
          bandBroadcastDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                                  static_cast<unsigned long long>(ctx.n);
          broadcastBandSuffix(ctx, dBand, 0, owner);
        } else if (segmentedTileRangeBcast) {
          const long cols = std::max<long>(0, bandEnd - bandStart);
          maxBandNeighborCols = std::max(maxBandNeighborCols, cols);
          const unsigned long long payloadDoubles =
              static_cast<unsigned long long>(2 * ctx.b) *
              static_cast<unsigned long long>(cols);
          if (useTilePullHandoff) {
            // The next consumer pulls exactly the window it needs from tileBandLatestOwners.
          } else if (segmentedTileRangeHandoff && !segmentedTileSweepOwner) {
            int successorOwner = -1;
            if (segmentedTileBatchMajor) {
              if (tile + 1 < tileCount) {
                successorOwner =
                    ownerOfColumn(std::min<long>(ctx.n - 1, rowTileEnd),
                                  ctx.counts,
                                  ctx.displs);
              } else if (batchEnd < totalSweeps) {
                successorOwner = ownerOfColumn(0, ctx.counts, ctx.displs);
              }
            } else {
              const long nextTileStart = rowTileEnd;
              if (nextTileStart < ctx.n) {
                successorOwner = ownerOfColumn(nextTileStart, ctx.counts, ctx.displs);
              }
            }
            if (successorOwner >= 0 && successorOwner != owner) {
              const bool wrapToNextBatch =
                  segmentedTileBatchMajor && tile + 1 == tileCount && batchEnd < totalSweeps;
              if (segmentedTileWrapGather && wrapToNextBatch) {
                const long nextBatchSweeps =
                    std::min<long>(tileBatchSweeps, totalSweeps - batchEnd);
                const long nextHaloCols =
                    2 * ctx.b * std::max<long>(0, nextBatchSweeps - 1) + ctx.b;
                const long nextBandStart = 0;
                const long nextBandEnd = std::min<long>(ctx.n, tileRows + nextHaloCols);
                bandWrapGatherDoubles +=
                    handoffRowTileWrapRangeToOwner(ctx,
                                                   dBand,
                                                   nextBandStart,
                                                   nextBandEnd,
                                                   tileRows,
                                                   tileCount,
                                                   haloCols,
                                                   successorOwner,
                                                   &maxBandNeighborCols);
              } else {
                if (cols > 0) {
                  bandNeighborDoubles += payloadDoubles;
                }
                handoffBandRangeToPeer(ctx, dBand, bandStart, bandEnd, owner, successorOwner);
              }
            }
          } else if (segmentedTileSuffixBcast) {
            bandSuffixBroadcastDoubles += payloadDoubles;
            broadcastBandColumnRangeSuffix(ctx, dBand, bandStart, bandEnd, owner);
          } else {
            bandBroadcastDoubles += payloadDoubles;
            broadcastBandColumnRange(ctx, dBand, bandStart, bandEnd, owner);
          }
        } else if (segmentedTileOwnerHandoff) {
          if (ownerHandoffBoundary) {
              const long bandStart = 0;
              const long bandEnd =
                  segmentedTileOwnerHandoffPrefix
                      ? std::min<long>(ctx.n, ctx.displs[nextBatchOwner] + brownCols)
                      : ctx.n;
              const long cols = bandEnd - bandStart;
              maxBandNeighborCols = std::max(maxBandNeighborCols, cols);
              bandOwnerHandoffDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                                         static_cast<unsigned long long>(cols);
              handoffBandRangeToPeer(ctx, dBand, bandStart, bandEnd, owner, nextBatchOwner);
          }
        } else {
          const long nextTileStart = rowTileEnd;
          if (nextTileStart < ctx.n) {
            const int nextOwner = ownerOfColumn(nextTileStart, ctx.counts, ctx.displs);
            if (nextOwner == owner + 1) {
              const long handoffStart = nextTileStart;
              const long handoffEnd = std::min<long>(ctx.n, handoffStart + brownCols);
              const long cols = std::max<long>(0, handoffEnd - handoffStart);
              maxBandNeighborCols = std::max(maxBandNeighborCols, cols);
              if (cols > 0) {
                bandNeighborDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                                        static_cast<unsigned long long>(cols);
              }
              handoffBandRangeToNextOwner(ctx, dBand, handoffStart, handoffEnd, owner);
            }
          }
          if (owner > 0) {
            const long backStart = ctx.displs[owner];
            const long backEnd = std::min<long>(ctx.n, backStart + brownCols);
            const long cols = std::max<long>(0, backEnd - backStart);
            if (cols > 0) {
              bandNeighborBackDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                                         static_cast<unsigned long long>(cols);
            }
            handoffBandRangeToPrevOwner(ctx, dBand, backStart, backEnd, owner);
          }
        }
	      }
	    }
	    if (segmentedTileRangeHandoff && !segmentedTileSweepOwner &&
	        segmentedTileBatchMajor && tileCount > 0 && batchCount > 0) {
	      if (useTilePullHandoff) {
	        for (int target = 0; target < ctx.size; ++target) {
	          const long ownedStart = ctx.displs[target];
	          const long ownedEnd = ctx.displs[target] + ctx.counts[target];
	          bandOwnerHandoffDoubles +=
	              pullBandRangeToOwner(ctx,
	                                   dBand,
	                                   ownedStart,
		                                   ownedEnd,
		                                   target,
		                                   tileBandLatestOwners,
		                                   &maxBandNeighborCols,
		                                   &bandPullCallCount,
		                                   &bandPullTransferCount,
		                                   &bandPullMaxTransfersPerCall);
	          markBandRangeOwner(ctx.n, ownedStart, ownedEnd, target, &tileBandLatestOwners);
	        }
	      } else {
	        const long finalBatchStart = (batchCount - 1) * tileBatchSweeps;
	        const long finalBatchEnd =
	            std::min<long>(totalSweeps, finalBatchStart + tileBatchSweeps);
	        const long finalBatchSweeps = std::max<long>(0, finalBatchEnd - finalBatchStart);
	        const long finalHaloCols =
	            2 * ctx.b * std::max<long>(0, finalBatchSweeps - 1) + ctx.b;
	        const long finalTileStart = (tileCount - 1) * tileRows;
	        const int finalOwner =
	            ownerOfColumn(std::min<long>(ctx.n - 1, finalTileStart),
	                          ctx.counts,
	                          ctx.displs);
	        for (int target = 0; target < ctx.size; ++target) {
	          const long ownedStart = ctx.displs[target];
	          const long ownedEnd = ctx.displs[target] + ctx.counts[target];
	          const long cols = std::max<long>(0, ownedEnd - ownedStart);
	          if (cols <= 0) {
	            continue;
	          }
	          if (segmentedTileWrapGather) {
	            bandOwnerHandoffDoubles +=
	                handoffRowTileWrapRangeToOwner(ctx,
	                                               dBand,
	                                               ownedStart,
	                                               ownedEnd,
	                                               tileRows,
	                                               tileCount,
	                                               finalHaloCols,
	                                               target,
	                                               &maxBandNeighborCols);
	          } else {
	            if (target == finalOwner) {
	              continue;
	            }
	            maxBandNeighborCols = std::max(maxBandNeighborCols, cols);
	            bandOwnerHandoffDoubles += static_cast<unsigned long long>(2 * ctx.b) *
	                                       static_cast<unsigned long long>(cols);
	            handoffBandRangeToPeer(ctx, dBand, ownedStart, ownedEnd, finalOwner, target);
	          }
	        }
	      }
	    }
	    if (dTileProgressPackedUCompact != nullptr) {
	      EVD_CUDA_CHECK(evdFree(dTileProgressPackedUCompact));
	    }
