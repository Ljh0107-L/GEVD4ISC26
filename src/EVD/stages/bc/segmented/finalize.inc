  if (segmentedBlockLocal) {
    // block-local broadcasts each owner's packedU slice as soon as that owner finishes.
  } else if (segmentedTileWave && segmentedTileSweepOwner) {
    for (int owner = 0; owner < ctx.size; ++owner) {
      const PackedUSlicePlan &plan = tileSweepOwnerPackedUPlans[owner];
      if (plan.totalElems <= 0) {
        continue;
      }
      packedUBroadcastDoubles += static_cast<unsigned long long>(plan.totalElems);
      broadcastPackedUOwnerSlicesCompact(ctx,
                                         dUPacked,
                                         dPackedUCompactChain,
                                         plan,
                                         owner);
    }
  } else if (segmentedTileWave && segmentedTileRangeHandoff && !segmentedTileSweepOwner &&
             segmentedTileRowOwnerPackedUBcast) {
    for (int owner = 0; owner < ctx.size; ++owner) {
      const PackedUProgressPlan &plan = tileRowOwnerPackedUPlans[owner];
      if (plan.totalElems <= 0) {
        continue;
	      }
	      packedUBroadcastDoubles += static_cast<unsigned long long>(plan.totalElems);
	      recordPackedUProgressPlan(plan);
	      broadcastPackedUProgressCompact(ctx,
	                                      dUPacked,
	                                      dPackedUCompactChain,
                                      plan,
                                      dPackedUOffsets,
                                      packedUSweepCount,
                                      owner,
                                      &packedUProgressWorkspace);
    }
  } else if (segmentedTileWave && segmentedTileProgressPackedUBcast) {
    if (!segmentedTileProgressPackedUImmediate) {
      for (int owner = 0; owner < ctx.size; ++owner) {
        PackedUProgressPlan plan;
        if (ctx.rank == owner &&
            static_cast<size_t>(owner) < tileProgressPackedUPlans.size()) {
          coalescePackedUProgressPlan(
              &tileProgressPackedUPlans[static_cast<size_t>(owner)]);
          plan = tileProgressPackedUPlans[static_cast<size_t>(owner)];
        }
        broadcastPackedUProgressPlanMetadata(ctx, &plan, owner);
        if (plan.totalElems <= 0) {
          continue;
        }
        if (dPackedUCompactChain == nullptr ||
            plan.totalElems > maxCumulativePackedUPlanElems) {
          if (dPackedUCompactChain != nullptr) {
            EVD_CUDA_CHECK(evdFree(dPackedUCompactChain));
          }
          EVD_CUDA_CHECK(evdMalloc(&dPackedUCompactChain,
                           sizeof(double) * static_cast<size_t>(plan.totalElems)));
          maxCumulativePackedUPlanElems = plan.totalElems;
	        }
	        packedUBroadcastDoubles += static_cast<unsigned long long>(plan.totalElems);
	        recordPackedUProgressPlan(plan);
	        broadcastPackedUProgressCompact(ctx,
	                                        dUPacked,
	                                        dPackedUCompactChain,
                                        plan,
                                        dPackedUOffsets,
                                        packedUSweepCount,
                                        owner,
                                        &packedUProgressWorkspace);
      }
    }
  } else if (segmentedTileWave || segmentedRowPipeline) {
    packedUAllreduceDoubles += static_cast<unsigned long long>(packedUElems);
    allReducePackedUInChunks(ctx, dUPacked, packedUElems);
  } else if (segmentedOwnerSerialPackedUOwnerBcast) {
    // owner-serial progress mode broadcasts packedU immediately after each owner.
  } else if (segmentedPackedUChain) {
    packedUBroadcastDoubles += static_cast<unsigned long long>(packedUElems);
    NCCL_CHECK_LOCAL(ncclBroadcast(dUPacked,
                                   dUPacked,
                                   static_cast<size_t>(packedUElems),
                                   ncclDouble,
                                   ctx.size - 1,
                                   ctx.nccl,
                                   ctx.commStream));
    EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
  } else {
    packedUAllreduceDoubles += static_cast<unsigned long long>(packedUElems);
    allReducePackedUInChunks(ctx, dUPacked, packedUElems);
  }
  if (ctx.rank == 0) {
    std::cout << "BC segmented comm estimate: band_bcast="
              << static_cast<double>(bandBroadcastDoubles * sizeof(double)) / (1024.0 * 1024.0)
              << " MiB, band_suffix_bcast="
              << static_cast<double>(bandSuffixBroadcastDoubles * sizeof(double)) /
                     (1024.0 * 1024.0)
              << " MiB, band_owner_handoff="
              << static_cast<double>(bandOwnerHandoffDoubles * sizeof(double)) /
                     (1024.0 * 1024.0)
              << " MiB, band_neighbor="
              << static_cast<double>(bandNeighborDoubles * sizeof(double)) / (1024.0 * 1024.0)
              << " MiB, band_neighbor_back="
              << static_cast<double>(bandNeighborBackDoubles * sizeof(double)) / (1024.0 * 1024.0)
              << " MiB, band_wrap_gather="
              << static_cast<double>(bandWrapGatherDoubles * sizeof(double)) / (1024.0 * 1024.0)
	              << " MiB, band_pull_handoff="
	              << static_cast<double>(bandPullHandoffDoubles * sizeof(double)) / (1024.0 * 1024.0)
		              << " MiB, band_pull_calls=" << bandPullCallCount
		              << ", band_pull_segments=" << bandPullTransferCount
		              << ", band_pull_max_segments_per_call=" << bandPullMaxTransfersPerCall
		              << ", band_preowner_handoff="
              << static_cast<double>(bandPreownerHandoffDoubles * sizeof(double)) /
                     (1024.0 * 1024.0)
              << " MiB, packedU_neighbor="
              << static_cast<double>(packedUNeighborDoubles * sizeof(double)) / (1024.0 * 1024.0 * 1024.0)
              << " GiB, packedU_bcast="
              << static_cast<double>(packedUBroadcastDoubles * sizeof(double)) / (1024.0 * 1024.0 * 1024.0)
              << " GiB, packedU_allreduce_payload="
              << static_cast<double>(packedUAllreduceDoubles * sizeof(double)) /
                     (1024.0 * 1024.0 * 1024.0)
              << " GiB, brown_cols=" << brownCols
              << ", full_band_sync=" << (segmentedFullBandSync ? 1 : 0)
              << ", brown_only=" << (segmentedBrownOnly ? 1 : 0)
              << ", flat_stop=" << (segmentedFlatStop ? 1 : 0)
              << ", flat_brown=" << (flatStopUsesBrown ? 1 : 0)
              << ", rolling=" << (segmentedRolling ? 1 : 0)
              << ", rolling_back_sync=" << (segmentedRollingBackSync ? 1 : 0)
	              << ", rolling_flat_stop=" << (segmentedRollingFlatStop ? 1 : 0)
	              << ", rolling_block_stop=" << (segmentedRollingBlockStop ? 1 : 0)
	              << ", rolling_frontier_window=" << (segmentedRollingFrontierWindow ? 1 : 0)
	              << ", rolling_full_bcast=" << (segmentedRollingFullBcast ? 1 : 0)
              << ", rolling_full_com_bcast=" << (segmentedRollingFullComBcast ? 1 : 0)
	              << ", rolling_dirty_handoff=" << (segmentedRollingDirtyHandoff ? 1 : 0)
	              << ", rolling_dirty_bcast=" << (segmentedRollingDirtyBcast ? 1 : 0)
	              << ", rolling_dirty_prefix=" << (segmentedRollingDirtyPrefix ? 1 : 0)
	              << ", rolling_pull_handoff=" << (segmentedRollingPullHandoff ? 1 : 0)
	              << ", row_pipeline=" << (segmentedRowPipeline ? 1 : 0)
              << ", row_pipeline_full_bcast=" << (segmentedRowPipelineFullBcast ? 1 : 0)
              << ", tile_wave=" << (segmentedTileWave ? 1 : 0)
              << ", tile_full_bcast=" << (segmentedTileWaveFullBcast ? 1 : 0)
              << ", tile_batch_major=" << (segmentedTileBatchMajor ? 1 : 0)
              << ", tile_stop_mode=" << segmentedTileStopMode
              << ", tile_single_step=" << (segmentedTileSingleStep ? 1 : 0)
              << ", tile_multi_step=" << (segmentedTileMultiStep ? 1 : 0)
              << ", tile_sweep_owner=" << (segmentedTileSweepOwner ? 1 : 0)
              << ", tile_range_bcast=" << (segmentedTileRangeBcast ? 1 : 0)
              << ", tile_range_handoff=" << (segmentedTileRangeHandoff ? 1 : 0)
              << ", tile_suffix_bcast=" << (segmentedTileSuffixBcast ? 1 : 0)
              << ", tile_owner_handoff=" << (segmentedTileOwnerHandoff ? 1 : 0)
              << ", tile_owner_handoff_prefix=" << (segmentedTileOwnerHandoffPrefix ? 1 : 0)
              << ", tile_sparse_com_sync=" << (segmentedTileSparseComSync ? 1 : 0)
              << ", tile_row_owner_packedu_bcast="
              << (segmentedTileRowOwnerPackedUBcast ? 1 : 0)
              << ", tile_progress_packedu_bcast="
              << (segmentedTileProgressPackedUBcast ? 1 : 0)
              << ", tile_progress_packedu_immediate="
              << (segmentedTileProgressPackedUImmediate ? 1 : 0)
              << ", tile_frontier_range=" << (segmentedTileFrontierRange ? 1 : 0)
              << ", tile_wrap_gather=" << (segmentedTileWrapGather ? 1 : 0)
              << ", tile_pull_handoff=" << (segmentedTilePullHandoff ? 1 : 0)
              << ", tile_preowner_band="
              << (envIntOrDefault("EVD_BC_SEGMENTED_TILE_PREOWNER_BAND", 0) != 0 ? 1 : 0)
              << ", tile_peer_com_handoff="
              << (envIntOrDefault("EVD_BC_SEGMENTED_TILE_PEER_COM_HANDOFF",
                                  segmentedTilePullHandoff ? 1 : 0) != 0 ? 1 : 0)
              << ", owner_gather_back=" << (segmentedOwnerGatherBack ? 1 : 0)
              << ", block_local=" << (segmentedBlockLocal ? 1 : 0)
              << ", block_local_full_bcast=" << (segmentedBlockLocalFullBcast ? 1 : 0)
              << ", block_local_slanted=" << (segmentedBlockLocalSlanted ? 1 : 0)
              << ", block_local_single_step=" << (segmentedBlockLocalSingleStep ? 1 : 0)
              << ", com_bcast="
              << static_cast<double>(comBroadcastInts * sizeof(int)) / (1024.0 * 1024.0)
              << " MiB"
              << ", com_owner_handoff="
              << static_cast<double>(comOwnerHandoffInts * sizeof(int)) / (1024.0 * 1024.0)
              << " MiB"
              << ", max_band_neighbor_cols=" << maxBandNeighborCols
              << ", full_band_neighbor_windows=" << fullBandNeighborWindows
              << ", packedu_chain=" << (segmentedPackedUChain ? 1 : 0)
              << ", packedu_compact_chain=" << (segmentedPackedUCompactChain ? 1 : 0)
              << ", packedu_owner_bcast="
              << (segmentedOwnerSerialPackedUOwnerBcast ? 1 : 0)
	              << ", packedu_compact_chain_max="
		              << static_cast<double>(maxCumulativePackedUPlanElems * sizeof(double)) /
		                     (1024.0 * 1024.0 * 1024.0)
		              << " GiB"
	              << ", packedu_progress_calls=" << packedUProgressBroadcastCalls
	              << ", packedu_progress_segments=" << packedUProgressSegmentCount
	              << ", packedu_progress_max_segments=" << packedUProgressMaxSegments
	              << ", packedu_progress_max_payload="
	              << static_cast<double>(packedUProgressMaxDoubles * sizeof(double)) /
	                     (1024.0 * 1024.0 * 1024.0)
	              << " GiB"
		              << ", com_chain=" << (segmentedComChain ? 1 : 0)
		              << ", pull_grouped=" << (segmentedPullGrouped ? 1 : 0)
	              << ", batch_sweeps=" << requestedBatchSweeps
	              << std::endl;
  }

  if (dPackedUCompactChain != nullptr) {
    EVD_CUDA_CHECK(evdFree(dPackedUCompactChain));
  }
  if (dBandBeforeOwner != nullptr) {
    EVD_CUDA_CHECK(evdFree(dBandBeforeOwner));
  }
  releasePackedUProgressDeviceWorkspace(&packedUProgressWorkspace);
  EVD_CUDA_CHECK(evdFree(com));
}
