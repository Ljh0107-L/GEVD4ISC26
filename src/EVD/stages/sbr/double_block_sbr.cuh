void broadcastBandColumnRange(const DistContext &ctx,
                              double *dBand,
                              long syncColStart,
                              long syncColEnd,
                              int owner);

void copySbrTrailingSnapshot(const DistContext &ctx,
                             long blockStart,
                             const double *dA,
                             double *dSnapshot)
{
  const long trailingStart = blockStart + ctx.b;
  const long firstLocalColumn =
      std::max(0L, std::min(ctx.localCols, trailingStart - ctx.colStart));
  const long columns = ctx.localCols - firstLocalColumn;
  const long rows = ctx.n - trailingStart;
  if (rows <= 0 || columns <= 0) {
    return;
  }

  const size_t pitchBytes = static_cast<size_t>(ctx.n) * sizeof(double);
  const size_t widthBytes = static_cast<size_t>(rows) * sizeof(double);
  const size_t offset = static_cast<size_t>(trailingStart) +
                        static_cast<size_t>(firstLocalColumn) * ctx.n;
  EVD_CUDA_CHECK(cudaMemcpy2D(dSnapshot + offset,
                     pitchBytes,
                     dA + offset,
                     pitchBytes,
                     widthBytes,
                     static_cast<size_t>(columns),
                     cudaMemcpyDeviceToDevice));
}

void packAndBroadcastCompletedBandBlock(const DistContext &ctx,
                                        double *dA,
                                        double *dBand,
                                        long blockStart,
                                        long blockEnd,
                                        int commFirstRank);

void distributedSbrDoubleBlock(const DistContext &ctx,
                               cusolverDnHandle_t cusolver,
                               cublasHandle_t cublas,
                               double *dA,
                               double *dWLocal,
                               double *dYLocal,
                               SbrWorkspace *ws,
                               bool useSuffixComm,
                               double *dBandPipeline = nullptr,
                               bool useBandPipeline = false,
                               bool pipelineEarlyReturn = false,
                               long pipelineGuardCols = 0)
{
  const double one = 1.0;
  const double zero = 0.0;
  const double negOne = -1.0;
  const double negHalf = -0.5;
  int *dInfo = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dInfo, sizeof(int)));
  SbrStats sbrStats;
  const bool debugStageProgress = debugStageProgressEnabled();
  if (debugStageProgress) {
    std::cout << "StageProgress rank=" << ctx.rank
              << " phase=sbr_double_begin local_cols=" << ctx.localCols
              << " col_start=" << ctx.colStart
              << " use_suffix_comm=" << (useSuffixComm ? 1 : 0)
              << " band_pipeline=" << (useBandPipeline ? 1 : 0)
              << " early_return=" << (pipelineEarlyReturn ? 1 : 0)
              << " guard_cols=" << pipelineGuardCols << std::endl;
  }

  for (long blockStart = 0; blockStart + ctx.b < ctx.n; blockStart += ctx.nb) {
    const long blockEnd = std::min(ctx.n, blockStart + ctx.nb);
    if (debugStageProgress) {
      std::cout << "StageProgress rank=" << ctx.rank
                << " phase=sbr_block_begin block=[" << blockStart << "," << blockEnd << ")"
                << std::endl;
    }
    copySbrTrailingSnapshot(ctx, blockStart, dA, ws->dABase);
    long blockColsGenerated = 0;

    for (long p = blockStart; p < blockEnd && p + ctx.b < ctx.n; p += ctx.b) {
      sbrStats.panels++;
      const long panelRowStart = p + ctx.b;
      const long panelRows = ctx.n - panelRowStart;
      const long blockOffset = p - blockStart;
      const long prevCols = blockOffset;
      const int owner = ownerOfColumn(p, ctx.counts, ctx.displs);
      const int commFirstRank = useSuffixComm ? owner : 0;
      const bool rankInSbrComm = ctx.rank >= commFirstRank;
      ncclComm_t sbrNccl = useSuffixComm ? ctx.sbrSuffixNccls[commFirstRank] : ctx.nccl;
      const int rootInComm = useSuffixComm ? 0 : owner;
      if (debugStageProgress) {
        std::cout << "StageProgress rank=" << ctx.rank
                  << " phase=sbr_panel_enter p=" << p
                  << " block_start=" << blockStart
                  << " owner=" << owner
                  << " comm_first_rank=" << commFirstRank
                  << " rank_in_comm=" << (rankInSbrComm ? 1 : 0)
                  << std::endl;
      }

      if (ctx.rank == owner) {
        const long localPanelCol = p - ctx.colStart;
        double *dPanel = dA + panelRowStart + localPanelCol * ctx.n;
        double *dPanelW = dWLocal + panelRowStart + localPanelCol * ctx.n;
        double *dPanelY = dYLocal + panelRowStart + localPanelCol * ctx.n;

        factorPanelQr(cusolver,
                cublas,
                panelRows,
                ctx.b,
                dPanel,
                ctx.n,
                dPanelW,
                ctx.n,
                ws->dR,
                ctx.n,
                ws->dWork,
                dInfo);
        if (debugStageProgress) {
          std::cout << "StageProgress rank=" << ctx.rank
                    << " phase=sbr_panel_qr_done p=" << p
                    << " panel_rows=" << panelRows << std::endl;
        }

        dim3 block(32, 32);
        dim3 grid(ceilDiv(panelRows, block.x), ceilDiv(ctx.b, block.y));
        launchCopyMatrix(grid, block, panelRows, ctx.b, dPanel, ctx.n, dPanelY, ctx.n);
        launchExtractHouseholderVectors(grid,
                          block,
                          static_cast<int>(panelRows),
                          static_cast<int>(ctx.b),
                          ws->dR,
                          static_cast<int>(ctx.n),
                          dPanel,
                          static_cast<int>(ctx.n));
        EVD_CUDA_CHECK(cudaGetLastError());
        EVD_CUDA_CHECK(cudaDeviceSynchronize());
        if (debugStageProgress) {
          std::cout << "StageProgress rank=" << ctx.rank
                    << " phase=sbr_panel_wy_done p=" << p
                    << std::endl;
        }
      }

      if (rankInSbrComm) {
        sbrStats.wyBroadcastDoubles +=
            static_cast<unsigned long long>(2 * ctx.b) *
            static_cast<unsigned long long>(panelRows);
        if (debugStageProgress) {
          std::cout << "StageProgress rank=" << ctx.rank
                    << " phase=sbr_panel_wy_bcast_begin p=" << p
                    << " owner=" << owner
                    << " root_in_comm=" << rootInComm << std::endl;
        }
        if (ctx.rank == owner) {
          const long localPanelCol = p - ctx.colStart;
          broadcastWYPanelsDevice(ctx,
                                  sbrNccl,
                                  dWLocal + panelRowStart + localPanelCol * ctx.n,
                                  dYLocal + panelRowStart + localPanelCol * ctx.n,
                                  ctx.n,
                                  ws->dWPanel,
                                  ws->dYPanel,
                                  panelRows,
                                  ctx.b,
                                  owner,
                                  rootInComm);
        } else {
          broadcastWYPanelsDevice(ctx,
                                  sbrNccl,
                                  nullptr,
                                  nullptr,
                                  ctx.n,
                                  ws->dWPanel,
                                  ws->dYPanel,
                                  panelRows,
                                  ctx.b,
                                  owner,
                                  rootInComm);
        }
        if (debugStageProgress) {
          std::cout << "StageProgress rank=" << ctx.rank
                    << " phase=sbr_panel_wy_bcast_done p=" << p
                    << " owner=" << owner << std::endl;
        }

        EVD_CUDA_CHECK(cudaMemcpy2D(ws->dYBlock + panelRowStart + blockOffset * ctx.n,
                           ctx.n * sizeof(double),
                           ws->dYPanel,
                           panelRows * sizeof(double),
                           panelRows * sizeof(double),
                           ctx.b,
                           cudaMemcpyDeviceToDevice));
        dim3 ybtBlock(32, 8);
        dim3 ybtGrid(ceilDiv(panelRows, ybtBlock.x), ceilDiv(ctx.b, ybtBlock.y));
        launchTransposeMatrix(ybtGrid,
                                         ybtBlock,
                                         panelRows,
                                         ctx.b,
                                         ws->dYPanel,
                                         panelRows,
                                         ws->dYBlockBT + blockOffset + panelRowStart * ctx.nb,
                                         ctx.nb);
        EVD_CUDA_CHECK(cudaGetLastError());
      }

      const long activeLocalStart = std::max(panelRowStart, ctx.colStart) - ctx.colStart;
      const long activeCols = (ctx.colStart + ctx.localCols > panelRowStart)
                                  ? (ctx.localCols - std::max(0L, activeLocalStart))
                                  : 0;
      const long activeGlobalStart = ctx.colStart + std::max(0L, activeLocalStart);

      if (activeCols > 0) {
        sbrStats.activePanels++;
        sbrStats.localUpdateEntries +=
            static_cast<unsigned long long>(panelRows) *
            static_cast<unsigned long long>(activeCols);
        dim3 gatherBlock(32, 8);
        dim3 gatherGrid(ceilDiv(ctx.b, gatherBlock.x), ceilDiv(activeCols, gatherBlock.y));
        gatherPanelRowsBT<<<gatherGrid, gatherBlock>>>(ws->dWPanel,
                                                       panelRows,
                                                       panelRowStart,
                                                       activeGlobalStart,
                                                       activeCols,
                                                       ws->dWBT,
                                                       ctx.b);
        gatherPanelRowsBT<<<gatherGrid, gatherBlock>>>(ws->dYPanel,
                                                       panelRows,
                                                       panelRowStart,
                                                       activeGlobalStart,
                                                       activeCols,
                                                       ws->dYBT,
                                                       ctx.b);
        EVD_CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       static_cast<int>(ctx.b),
                                       static_cast<int>(activeCols),
                                       static_cast<int>(panelRows),
                                       &one,
                                       ws->dWPanel,
                                       static_cast<int>(panelRows),
                                       ws->dABase + panelRowStart + std::max(0L, activeLocalStart) * ctx.n,
                                       static_cast<int>(ctx.n),
                                       &zero,
                                       ws->dAWBT,
                                       static_cast<int>(ctx.b)));

        if (prevCols > 0) {
          CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                         CUBLAS_OP_N,
                                         CUBLAS_OP_N,
                                         static_cast<int>(prevCols),
                                         static_cast<int>(ctx.b),
                                         static_cast<int>(panelRows),
                                         &one,
                                         ws->dZBlockBT + panelRowStart * ctx.nb,
                                         static_cast<int>(ctx.nb),
                                         ws->dWPanel,
                                         static_cast<int>(panelRows),
                                         &zero,
                                         ws->dTmpPrev1,
                                         static_cast<int>(prevCols)));
          CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_T,
                                         static_cast<int>(ctx.b),
                                         static_cast<int>(activeCols),
                                         static_cast<int>(prevCols),
                                         &negOne,
                                         ws->dTmpPrev1,
                                         static_cast<int>(prevCols),
                                         ws->dYBlock + activeGlobalStart,
                                         static_cast<int>(ctx.n),
                                         &one,
                                         ws->dAWBT,
                                         static_cast<int>(ctx.b)));
          CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         static_cast<int>(prevCols),
                                         static_cast<int>(ctx.b),
                                         static_cast<int>(panelRows),
                                         &one,
                                         ws->dYBlock + panelRowStart,
                                         static_cast<int>(ctx.n),
                                         ws->dWPanel,
                                         static_cast<int>(panelRows),
                                         &zero,
                                         ws->dTmpPrev2,
                                         static_cast<int>(prevCols)));
          CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         static_cast<int>(ctx.b),
                                         static_cast<int>(activeCols),
                                         static_cast<int>(prevCols),
                                         &negOne,
                                         ws->dTmpPrev2,
                                         static_cast<int>(prevCols),
                                         ws->dZBlockBT + activeGlobalStart * ctx.nb,
                                         static_cast<int>(ctx.nb),
                                         &one,
                                         ws->dAWBT,
                                         static_cast<int>(ctx.b)));
        }

        CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_T,
                                       static_cast<int>(ctx.b),
                                       static_cast<int>(ctx.b),
                                       static_cast<int>(activeCols),
                                       &one,
                                       ws->dWBT,
                                       static_cast<int>(ctx.b),
                                       ws->dAWBT,
                                       static_cast<int>(ctx.b),
                                       &zero,
                                       ws->dSLocal,
                                       static_cast<int>(ctx.b)));
      } else if (rankInSbrComm) {
        EVD_CUDA_CHECK(cudaMemset(ws->dSLocal, 0, sizeof(double) * ctx.b * ctx.b));
      }

      if (rankInSbrComm) {
        EVD_CUDA_CHECK(cudaDeviceSynchronize());
        sbrStats.allReduceDoubles +=
            static_cast<unsigned long long>(ctx.b) *
            static_cast<unsigned long long>(ctx.b);
        NCCL_CHECK_LOCAL(ncclAllReduce(ws->dSLocal,
                                       ws->dS,
                                       static_cast<size_t>(ctx.b * ctx.b),
                                       ncclDouble,
                                       ncclSum,
                                       sbrNccl,
                                       ctx.commStream));
        EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
      }

      if (activeCols > 0) {
        EVD_CUDA_CHECK(cudaMemcpy(ws->dZBT, ws->dAWBT, sizeof(double) * ctx.b * activeCols, cudaMemcpyDeviceToDevice));
        CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       static_cast<int>(ctx.b),
                                       static_cast<int>(activeCols),
                                       static_cast<int>(ctx.b),
                                       &negHalf,
                                       ws->dS,
                                       static_cast<int>(ctx.b),
                                       ws->dYBT,
                                       static_cast<int>(ctx.b),
                                       &one,
                                       ws->dZBT,
                                       static_cast<int>(ctx.b)));
      }

      if (rankInSbrComm) {
        for (int peer = commFirstRank; peer < ctx.size; ++peer) {
          const long peerStart = ctx.displs[peer];
          const long peerEnd = peerStart + ctx.counts[peer];
          const long sliceStart = std::max(panelRowStart, peerStart);
          if (peerEnd > sliceStart) {
            sbrStats.zBroadcastDoubles +=
                static_cast<unsigned long long>(ctx.b) *
                static_cast<unsigned long long>(peerEnd - sliceStart);
          }
        }
        broadcastZActiveSlices(ctx,
                               sbrNccl,
                               commFirstRank,
                               ws->dZBT,
                               activeCols,
                               activeGlobalStart,
                               ws->dZFullBT,
                               panelRowStart);
        if (debugStageProgress) {
          std::cout << "StageProgress rank=" << ctx.rank
                    << " phase=sbr_panel_comm_done p=" << p
                    << " owner=" << owner
                    << " active_cols=" << activeCols << std::endl;
        }
        EVD_CUDA_CHECK(cudaMemcpy2D(ws->dZBlockBT + blockOffset + panelRowStart * ctx.nb,
                           ctx.nb * sizeof(double),
                           ws->dZFullBT + panelRowStart * ctx.b,
                           ctx.b * sizeof(double),
                           ctx.b * sizeof(double),
                           panelRows,
                           cudaMemcpyDeviceToDevice));
      }

      blockColsGenerated = blockOffset + ctx.b;
      const long nextPanelCol = p + ctx.b;
	      if (nextPanelCol < blockEnd && nextPanelCol < ctx.n) {
        const int nextOwner = ownerOfColumn(nextPanelCol, ctx.counts, ctx.displs);
        if (ctx.rank == nextOwner) {
          const long localNextCol = nextPanelCol - ctx.colStart;
          double *dANext = dA + nextPanelCol + localNextCol * ctx.n;
          EVD_CUDA_CHECK(cudaMemcpy2D(dANext,
                             ctx.n * sizeof(double),
                             ws->dABase + nextPanelCol + localNextCol * ctx.n,
                             ctx.n * sizeof(double),
                             (ctx.n - nextPanelCol) * sizeof(double),
                             ctx.b,
                             cudaMemcpyDeviceToDevice));
          CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                         CUBLAS_OP_N,
                                         CUBLAS_OP_N,
                                         static_cast<int>(ctx.n - nextPanelCol),
                                         static_cast<int>(ctx.b),
                                         static_cast<int>(blockColsGenerated),
                                         &negOne,
                                         ws->dYBlock + nextPanelCol,
                                         static_cast<int>(ctx.n),
                                         ws->dZBlockBT + nextPanelCol * ctx.nb,
                                         static_cast<int>(ctx.nb),
                                         &one,
                                         dANext,
                                         static_cast<int>(ctx.n)));
          CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_T,
                                         static_cast<int>(ctx.n - nextPanelCol),
                                         static_cast<int>(ctx.b),
                                         static_cast<int>(blockColsGenerated),
                                         &negOne,
                                         ws->dZBlockBT + nextPanelCol * ctx.nb,
                                         static_cast<int>(ctx.nb),
                                         ws->dYBlock + nextPanelCol,
                                         static_cast<int>(ctx.n),
                                         &one,
                                         dANext,
                                         static_cast<int>(ctx.n)));
          sbrStats.updateGemmCalls += 2;
          sbrStats.localUpdateEntries +=
              static_cast<unsigned long long>(ctx.n - nextPanelCol) *
              static_cast<unsigned long long>(ctx.b);
        }
      }
    }

    if (blockEnd < ctx.n && blockColsGenerated > 0) {
      const bool usePanelwiseTailUpdate = envIntOrDefault("EVD_SBR_DOUBLE_TAIL_PANELWISE", 0) != 0;
      const long tailLocalStart = std::max(blockEnd, ctx.colStart) - ctx.colStart;
      const long tailCols = (ctx.colStart + ctx.localCols > blockEnd)
                                ? (ctx.localCols - std::max(0L, tailLocalStart))
                                : 0;
      const long tailGlobalStart = ctx.colStart + std::max(0L, tailLocalStart);
      if (tailCols > 0) {
        double *dATail = dA + blockEnd + std::max(0L, tailLocalStart) * ctx.n;
        EVD_CUDA_CHECK(cudaMemcpy2D(dATail,
                           ctx.n * sizeof(double),
                           ws->dABase + blockEnd + std::max(0L, tailLocalStart) * ctx.n,
                           ctx.n * sizeof(double),
                           (ctx.n - blockEnd) * sizeof(double),
                           tailCols,
                           cudaMemcpyDeviceToDevice));
        const bool localTailIsSquare =
            tailGlobalStart == blockEnd && tailCols == (ctx.n - blockEnd);
        if (!usePanelwiseTailUpdate && localTailIsSquare) {
          CUBLAS_CHECK_LOCAL(cublasDsyr2k(cublas,
                                          CUBLAS_FILL_MODE_LOWER,
                                          CUBLAS_OP_T,
                                          static_cast<int>(ctx.n - blockEnd),
                                          static_cast<int>(blockColsGenerated),
                                          &negOne,
                                          ws->dYBlockBT + blockEnd * ctx.nb,
                                          static_cast<int>(ctx.nb),
                                          ws->dZBlockBT + blockEnd * ctx.nb,
                                          static_cast<int>(ctx.nb),
                                          &one,
                                          dATail,
                                          static_cast<int>(ctx.n)));
          dim3 copyBlock(32, 32);
          dim3 copyGrid(ceilDiv(ctx.n - blockEnd, copyBlock.x),
                        ceilDiv(ctx.n - blockEnd, copyBlock.y));
          launchCopyLowerToUpper(copyGrid, copyBlock, ctx.n - blockEnd, dATail, ctx.n);
          EVD_CUDA_CHECK(cudaGetLastError());
          sbrStats.tailSyr2kCalls++;
        } else if (usePanelwiseTailUpdate) {
          for (long col = 0; col < blockColsGenerated; col += ctx.b) {
            const long curCols = std::min(ctx.b, blockColsGenerated - col);
            CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_N,
                                           static_cast<int>(ctx.n - blockEnd),
                                           static_cast<int>(tailCols),
                                           static_cast<int>(curCols),
                                           &negOne,
                                           ws->dYBlock + blockEnd + col * ctx.n,
                                           static_cast<int>(ctx.n),
                                           ws->dZBlockBT + col + tailGlobalStart * ctx.nb,
                                           static_cast<int>(ctx.nb),
                                           &one,
                                           dATail,
                                           static_cast<int>(ctx.n)));
            CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_T,
                                           static_cast<int>(ctx.n - blockEnd),
                                           static_cast<int>(tailCols),
                                           static_cast<int>(curCols),
                                           &negOne,
                                           ws->dZBlockBT + col + blockEnd * ctx.nb,
                                           static_cast<int>(ctx.nb),
                                           ws->dYBlock + tailGlobalStart + col * ctx.n,
                                           static_cast<int>(ctx.n),
                                           &one,
                                           dATail,
                                           static_cast<int>(ctx.n)));
            sbrStats.updateGemmCalls += 2;
          }
        } else {
          CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                         CUBLAS_OP_N,
                                         CUBLAS_OP_N,
                                         static_cast<int>(ctx.n - blockEnd),
                                         static_cast<int>(tailCols),
                                         static_cast<int>(blockColsGenerated),
                                         &negOne,
                                         ws->dYBlock + blockEnd,
                                         static_cast<int>(ctx.n),
                                         ws->dZBlockBT + tailGlobalStart * ctx.nb,
                                         static_cast<int>(ctx.nb),
                                         &one,
                                         dATail,
                                         static_cast<int>(ctx.n)));
          CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_T,
                                         static_cast<int>(ctx.n - blockEnd),
                                         static_cast<int>(tailCols),
                                         static_cast<int>(blockColsGenerated),
                                         &negOne,
                                         ws->dZBlockBT + blockEnd * ctx.nb,
                                         static_cast<int>(ctx.nb),
                                         ws->dYBlock + tailGlobalStart,
                                         static_cast<int>(ctx.n),
                                         &one,
                                         dATail,
                                         static_cast<int>(ctx.n)));
          sbrStats.updateGemmCalls += 2;
        }
        sbrStats.localUpdateEntries +=
            static_cast<unsigned long long>(ctx.n - blockEnd) *
            static_cast<unsigned long long>(tailCols);
      }
    }

    if (useBandPipeline && dBandPipeline != nullptr) {
      const long guardedBlockStart =
          pipelineEarlyReturn ? std::max<long>(0, blockStart - pipelineGuardCols) : 0;
      const int bandCommFirstRank =
          pipelineEarlyReturn ? ownerOfColumn(guardedBlockStart, ctx.counts, ctx.displs) : 0;
      if (debugStageProgress) {
        std::cout << "StageProgress rank=" << ctx.rank
                  << " phase=sbr_pack_band_begin block=[" << blockStart << "," << blockEnd << ")"
                  << " band_comm_first_rank=" << bandCommFirstRank
                  << std::endl;
      }
      packAndBroadcastCompletedBandBlock(ctx,
                                         dA,
                                         dBandPipeline,
                                         blockStart,
                                         blockEnd,
                                         bandCommFirstRank);
      if (debugStageProgress) {
        std::cout << "StageProgress rank=" << ctx.rank
                  << " phase=sbr_pack_band_done block=[" << blockStart << "," << blockEnd << ")"
                  << std::endl;
      }
    }

    if (debugStageProgress && ctx.rank == 0) {
      std::cout << "SBR double-block " << blockStart << "/" << ctx.n << std::endl;
    }

    const long earlyReturnEnd =
        std::min<long>(ctx.n, ctx.colStart + ctx.localCols + pipelineGuardCols);
    if (pipelineEarlyReturn && ctx.rank == 0 && blockEnd >= earlyReturnEnd) {
      if (debugStageProgress) {
        std::cout << "StageProgress rank=" << ctx.rank
                  << " phase=sbr_early_return block_end=" << blockEnd
                  << " early_return_end=" << earlyReturnEnd << std::endl;
      }
      if (printStageStatisticsEnabled()) {
        if (ctx.rank == 0) {
          std::cout << "SBR pipeline early-return enabled; per-rank SBR counters are printed locally"
                    << std::endl;
        }
        std::cout << "SBR double-block local counters rank=" << ctx.rank
                  << " panels=" << sbrStats.panels
                  << ", active_panels=" << sbrStats.activePanels
                  << ", wy_bcast="
                  << static_cast<double>(sbrStats.wyBroadcastDoubles * sizeof(double)) / (1024.0 * 1024.0)
                  << " MiB, z_bcast="
                  << static_cast<double>(sbrStats.zBroadcastDoubles * sizeof(double)) / (1024.0 * 1024.0)
                  << " MiB, allreduce="
                  << static_cast<double>(sbrStats.allReduceDoubles * sizeof(double)) / (1024.0 * 1024.0)
                  << " MiB, update_gemm_calls=" << sbrStats.updateGemmCalls
                  << ", tail_syr2k_calls=" << sbrStats.tailSyr2kCalls
                  << ", skipped_z_bcast_panels=" << sbrStats.skippedZBroadcastPanels
                  << ", local_update_entries=" << sbrStats.localUpdateEntries
                  << std::endl;
      }
      EVD_CUDA_CHECK(evdFree(dInfo));
      return;
    }
  }

  if (printStageStatisticsEnabled()) {
    unsigned long long localStatValues[] = {
        sbrStats.panels,
        sbrStats.activePanels,
        sbrStats.wyBroadcastDoubles,
        sbrStats.zBroadcastDoubles,
        sbrStats.allReduceDoubles,
        sbrStats.updateGemmCalls,
        sbrStats.tailSyr2kCalls,
        sbrStats.skippedZBroadcastPanels,
        sbrStats.localUpdateEntries,
    };
    unsigned long long sumStatValues[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    MPI_CHECK(MPI_Reduce(localStatValues,
                         sumStatValues,
                         9,
                         MPI_UNSIGNED_LONG_LONG,
                         MPI_SUM,
                         0,
                         ctx.comm));
    if (ctx.rank == 0) {
      std::cout << "SBR double-block counters: panels=" << sbrStats.panels
                << ", panels_rank_sum=" << sumStatValues[0]
                << ", active_panels_rank_sum=" << sumStatValues[1]
                << ", wy_bcast_rank_sum="
                << static_cast<double>(sumStatValues[2] * sizeof(double)) / (1024.0 * 1024.0)
                << " MiB, z_bcast_rank_sum="
                << static_cast<double>(sumStatValues[3] * sizeof(double)) / (1024.0 * 1024.0)
                << " MiB, allreduce_rank_sum="
                << static_cast<double>(sumStatValues[4] * sizeof(double)) / (1024.0 * 1024.0)
                << " MiB, update_gemm_calls_sum=" << sumStatValues[5]
                << ", tail_syr2k_calls_sum=" << sumStatValues[6]
                << ", skipped_z_bcast_panels_rank_sum=" << sumStatValues[7]
                << ", local_update_entries_sum=" << sumStatValues[8]
                << std::endl;
    }
  }

  EVD_CUDA_CHECK(evdFree(dInfo));
}

