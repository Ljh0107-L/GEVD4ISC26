void distributedSbr(const DistContext &ctx,
                    cusolverDnHandle_t cusolver,
                    cublasHandle_t cublas,
                    double *dA,
                    double *dWLocal,
                    double *dYLocal,
                    SbrWorkspace *ws,
                    bool useSuffixComm)
{
  const double one = 1.0;
  const double zero = 0.0;
  const double negOne = -1.0;
  const double negHalf = -0.5;
  int *dInfo = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dInfo, sizeof(int)));
  SbrStats sbrStats;

  for (long p = 0; p + ctx.b < ctx.n; p += ctx.b) {
    sbrStats.panels++;
    const long panelRowStart = p + ctx.b;
    const long panelRows = ctx.n - panelRowStart;
    const int owner = ownerOfColumn(p, ctx.counts, ctx.displs);
    const int commFirstRank = useSuffixComm ? owner : 0;
    const bool rankInSbrComm = ctx.rank >= commFirstRank;
    ncclComm_t sbrNccl = useSuffixComm ? ctx.sbrSuffixNccls[commFirstRank] : ctx.nccl;
    const int rootInComm = useSuffixComm ? 0 : owner;

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
    }

    if (rankInSbrComm && panelRows > 0) {
      sbrStats.wyBroadcastDoubles +=
          static_cast<unsigned long long>(2 * ctx.b) *
          static_cast<unsigned long long>(panelRows);
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
                                     dA + panelRowStart + std::max(0L, activeLocalStart) * ctx.n,
                                     static_cast<int>(ctx.n),
                                     &zero,
                                     ws->dAWBT,
                                     static_cast<int>(ctx.b)));

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
    const bool activeTailSingleRank = ownerOfColumn(panelRowStart, ctx.counts, ctx.displs) == ctx.size - 1;
    if (rankInSbrComm && !activeTailSingleRank) {
      for (int peer = 0; peer < ctx.size; ++peer) {
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
    } else if (rankInSbrComm) {
      sbrStats.skippedZBroadcastPanels++;
    }

    if (activeCols > 0) {
      double *dATail = dA + panelRowStart + std::max(0L, activeLocalStart) * ctx.n;
      const bool localTailIsSquare = activeTailSingleRank && activeGlobalStart == panelRowStart && activeCols == panelRows;
      if (localTailIsSquare) {
        sbrStats.tailSyr2kCalls++;
        CUBLAS_CHECK_LOCAL(cublasDsyr2k(cublas,
                                        CUBLAS_FILL_MODE_LOWER,
                                        CUBLAS_OP_T,
                                        static_cast<int>(panelRows),
                                        static_cast<int>(ctx.b),
                                        &negOne,
                                        ws->dYBT,
                                        static_cast<int>(ctx.b),
                                        ws->dZBT,
                                        static_cast<int>(ctx.b),
                                        &one,
                                        dATail,
                                        static_cast<int>(ctx.n)));
        dim3 copyBlock(32, 32);
        dim3 copyGrid(ceilDiv(panelRows, copyBlock.x), ceilDiv(panelRows, copyBlock.y));
        launchCopyLowerToUpper(copyGrid, copyBlock, panelRows, dATail, ctx.n);
        EVD_CUDA_CHECK(cudaGetLastError());
      } else {
        sbrStats.updateGemmCalls += 2;
        CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       static_cast<int>(panelRows),
                                       static_cast<int>(activeCols),
                                       static_cast<int>(ctx.b),
                                       &negOne,
                                       ws->dYPanel,
                                       static_cast<int>(panelRows),
                                       ws->dZBT,
                                       static_cast<int>(ctx.b),
                                       &one,
                                       dATail,
                                       static_cast<int>(ctx.n)));
        CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                       CUBLAS_OP_T,
                                       CUBLAS_OP_N,
                                       static_cast<int>(panelRows),
                                       static_cast<int>(activeCols),
                                       static_cast<int>(ctx.b),
                                       &negOne,
                                       ws->dZFullBT + panelRowStart * ctx.b,
                                       static_cast<int>(ctx.b),
                                       ws->dYBT,
                                       static_cast<int>(ctx.b),
                                       &one,
                                       dATail,
                                       static_cast<int>(ctx.n)));
      }
    }

    if (debugStageProgressEnabled() && ctx.rank == 0 && ((p / ctx.b) % 64 == 0)) {
      std::cout << "SBR panel " << p << "/" << ctx.n << std::endl;
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
      std::cout << "SBR comm/compute counters: panels=" << sbrStats.panels
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

