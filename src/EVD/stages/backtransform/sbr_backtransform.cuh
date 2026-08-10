void distributedSbrBack(const DistContext &ctx,
                        cublasHandle_t cublas,
                        double *dQ,
                        long ldQ,
                        double *dWLocal,
                        double *dYLocal,
                        SbrWorkspace *ws)
{
  const double one = 1.0;
  const double zero = 0.0;
  const double negOne = -1.0;
  double *dTmp = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dTmp, sizeof(double) * static_cast<size_t>(ctx.b) * ctx.qLocalCols));

  for (long p = ctx.n - 2 * ctx.b; p >= 0; p -= ctx.b) {
    const long panelRowStart = p + ctx.b;
    const long panelRows = ctx.n - panelRowStart;
    const int owner = ownerOfColumn(p, ctx.counts, ctx.displs);
    if (ctx.rank == owner) {
      const long localPanelCol = p - ctx.colStart;
      broadcastWYPanelsDevice(ctx,
                              ctx.nccl,
                              dWLocal + panelRowStart + localPanelCol * ctx.n,
                              dYLocal + panelRowStart + localPanelCol * ctx.n,
                              ctx.n,
                              ws->dWPanel,
                              ws->dYPanel,
                              panelRows,
                              ctx.b,
                              owner,
                              owner);
    } else {
      broadcastWYPanelsDevice(ctx,
                              ctx.nccl,
                              nullptr,
                              nullptr,
                              ctx.n,
                              ws->dWPanel,
                              ws->dYPanel,
                              panelRows,
                              ctx.b,
                              owner,
                              owner);
    }

    CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_N,
                                   static_cast<int>(ctx.b),
                                   static_cast<int>(ctx.qLocalCols),
                                   static_cast<int>(panelRows),
                                   &one,
                                   ws->dYPanel,
                                   static_cast<int>(panelRows),
                                   dQ + panelRowStart,
                                   static_cast<int>(ldQ),
                                   &zero,
                                   dTmp,
                                   static_cast<int>(ctx.b)));

    CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                   CUBLAS_OP_N,
                                   CUBLAS_OP_N,
                                   static_cast<int>(panelRows),
                                   static_cast<int>(ctx.qLocalCols),
                                   static_cast<int>(ctx.b),
                                   &negOne,
                                   ws->dWPanel,
                                   static_cast<int>(panelRows),
                                   dTmp,
                                   static_cast<int>(ctx.b),
                                   &one,
                                   dQ + panelRowStart,
                                   static_cast<int>(ldQ)));

    if (p == 0) {
      break;
    }
  }

  EVD_CUDA_CHECK(evdFree(dTmp));
}

void gatherSbrBlockWYToAll(const DistContext &ctx,
                           long blockStart,
                           long blockWidth,
                           double *dWLocal,
                           double *dYLocal,
                           SbrWorkspace *ws,
                           double *dWBlock,
                           double *dYBlock)
{
  EVD_CUDA_CHECK(cudaMemset(dWBlock, 0, sizeof(double) * static_cast<size_t>(ctx.n) * blockWidth));
  EVD_CUDA_CHECK(cudaMemset(dYBlock, 0, sizeof(double) * static_cast<size_t>(ctx.n) * blockWidth));

  const long blockEnd = blockStart + blockWidth;
  for (long p = blockStart; p < blockEnd && p + ctx.b < ctx.n; p += ctx.b) {
    const long panelRowStart = p + ctx.b;
    const long panelRows = ctx.n - panelRowStart;
    const long blockCol = p - blockStart;
    const int owner = ownerOfColumn(p, ctx.counts, ctx.displs);
    if (ctx.rank == owner) {
      const long localPanelCol = p - ctx.colStart;
      broadcastWYPanelsDevice(ctx,
                              ctx.nccl,
                              dWLocal + panelRowStart + localPanelCol * ctx.n,
                              dYLocal + panelRowStart + localPanelCol * ctx.n,
                              ctx.n,
                              ws->dWPanel,
                              ws->dYPanel,
                              panelRows,
                              ctx.b,
                              owner,
                              owner);
    } else {
      broadcastWYPanelsDevice(ctx,
                              ctx.nccl,
                              nullptr,
                              nullptr,
                              ctx.n,
                              ws->dWPanel,
                              ws->dYPanel,
                              panelRows,
                              ctx.b,
                              owner,
                              owner);
    }

    EVD_CUDA_CHECK(cudaMemcpy2D(dWBlock + panelRowStart + blockCol * ctx.n,
                       ctx.n * sizeof(double),
                       ws->dWPanel,
                       panelRows * sizeof(double),
                       panelRows * sizeof(double),
                       ctx.b,
                       cudaMemcpyDeviceToDevice));
    EVD_CUDA_CHECK(cudaMemcpy2D(dYBlock + panelRowStart + blockCol * ctx.n,
                       ctx.n * sizeof(double),
                       ws->dYPanel,
                       panelRows * sizeof(double),
                       panelRows * sizeof(double),
                       ctx.b,
                       cudaMemcpyDeviceToDevice));
  }
}

void composeSbrBlockW(const DistContext &ctx,
                      cublasHandle_t cublas,
                      long blockStart,
                      long blockWidth,
                      double *dWBlock,
                      double *dYBlock,
                      double *dTmp)
{
  const double one = 1.0;
  const double zero = 0.0;
  const double negOne = -1.0;
  const long rowStart = std::min(ctx.n, blockStart + ctx.b);
  const long rows = ctx.n - rowStart;
  if (rows <= 0) {
    return;
  }

  for (long colWidth = ctx.b; colWidth < blockWidth; colWidth *= 2) {
    for (long group = 0; group + colWidth < blockWidth; group += 2 * colWidth) {
      const long rightWidth = std::min(colWidth, blockWidth - group - colWidth);
      if (rightWidth <= 0) {
        continue;
      }
      CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     static_cast<int>(colWidth),
                                     static_cast<int>(rightWidth),
                                     static_cast<int>(rows),
                                     &one,
                                     dYBlock + rowStart + group * ctx.n,
                                     static_cast<int>(ctx.n),
                                     dWBlock + rowStart + (group + colWidth) * ctx.n,
                                     static_cast<int>(ctx.n),
                                     &zero,
                                     dTmp,
                                     static_cast<int>(blockWidth)));
      CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     static_cast<int>(rows),
                                     static_cast<int>(rightWidth),
                                     static_cast<int>(colWidth),
                                     &negOne,
                                     dWBlock + rowStart + group * ctx.n,
                                     static_cast<int>(ctx.n),
                                     dTmp,
                                     static_cast<int>(blockWidth),
                                     &one,
                                     dWBlock + rowStart + (group + colWidth) * ctx.n,
                                     static_cast<int>(ctx.n)));
    }
  }
}

void distributedSbrBackDoubleBlock(const DistContext &ctx,
                                   cublasHandle_t cublas,
                                   double *dQ,
                                   long ldQ,
                                   double *dWLocal,
                                   double *dYLocal,
                                   SbrWorkspace *ws)
{
  const bool printProgress = debugStageProgressEnabled();
  if (ws->dABase == nullptr || ws->dYBlock == nullptr || ws->dWork == nullptr) {
    if (ctx.rank == 0) {
      std::cerr << "double-block SBR-back requires double-block SBR workspace" << std::endl;
    }
    MPI_Abort(ctx.comm, 1);
  }

  const double one = 1.0;
  const double zero = 0.0;
  const double negOne = -1.0;
  double *dWBlock = ws->dABase;
  double *dYBlock = ws->dYBlock;
  double *dComposeTmp = ws->dWork;
  double *dApplyTmp = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dApplyTmp, sizeof(double) * static_cast<size_t>(ctx.nb) * ctx.qLocalCols));

  long lastBlockStart = ((ctx.n - 1) / ctx.nb) * ctx.nb;
  for (long blockStart = lastBlockStart; blockStart >= 0; blockStart -= ctx.nb) {
    const long blockWidth = std::min(ctx.nb, ctx.n - blockStart);
    if (blockWidth <= 0) {
      if (blockStart == 0) {
        break;
      }
      continue;
    }

    gatherSbrBlockWYToAll(ctx,
                          blockStart,
                          blockWidth,
                          dWLocal,
                          dYLocal,
                          ws,
                          dWBlock,
                          dYBlock);
    composeSbrBlockW(ctx, cublas, blockStart, blockWidth, dWBlock, dYBlock, dComposeTmp);

    const long rowStart = std::min(ctx.n, blockStart + ctx.b);
    const long rows = ctx.n - rowStart;
    if (rows > 0) {
      CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     static_cast<int>(blockWidth),
                                     static_cast<int>(ctx.qLocalCols),
                                     static_cast<int>(rows),
                                     &one,
                                     dYBlock + rowStart,
                                     static_cast<int>(ctx.n),
                                     dQ + rowStart,
                                     static_cast<int>(ldQ),
                                     &zero,
                                     dApplyTmp,
                                     static_cast<int>(blockWidth)));
      CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     static_cast<int>(rows),
                                     static_cast<int>(ctx.qLocalCols),
                                     static_cast<int>(blockWidth),
                                     &negOne,
                                     dWBlock + rowStart,
                                     static_cast<int>(ctx.n),
                                     dApplyTmp,
                                     static_cast<int>(blockWidth),
                                     &one,
                                     dQ + rowStart,
                                     static_cast<int>(ldQ)));
    }

    if (printProgress && ctx.rank == 0) {
      std::cout << "SBR double-block back block " << blockStart << "/" << ctx.n << std::endl;
    }
    if (blockStart == 0) {
      break;
    }
  }

  EVD_CUDA_CHECK(evdFree(dApplyTmp));
}

void transposeDistributedColumnMatrix(const DistContext &ctx,
                                      double *dInput,
                                      long ldInput,
                                      double *dOutput,
                                      long ldOutput)
{
  double *dSendPacked = dOutput;
  double *dRecvPacked = dInput;

  std::vector<size_t> sendOffsets(ctx.size, 0);
  std::vector<size_t> recvOffsets(ctx.size, 0);
  size_t sendOffset = 0;
  size_t recvOffset = 0;
  for (int r = 0; r < ctx.size; ++r) {
    sendOffsets[r] = sendOffset;
    recvOffsets[r] = recvOffset;
    sendOffset += static_cast<size_t>(ctx.qCounts[r]) * static_cast<size_t>(ctx.qLocalCols);
    recvOffset += static_cast<size_t>(ctx.qCounts[r]) * static_cast<size_t>(ctx.qLocalCols);
  }

  for (int dst = 0; dst < ctx.size; ++dst) {
    const long dstRows = ctx.qCounts[dst];
    const long dstRowStart = ctx.qDispls[dst];
    EVD_CUDA_CHECK(cudaMemcpy2DAsync(dSendPacked + sendOffsets[dst],
                            dstRows * sizeof(double),
                            dInput + dstRowStart,
                            ldInput * sizeof(double),
                            dstRows * sizeof(double),
                            ctx.qLocalCols,
                            cudaMemcpyDeviceToDevice,
                            ctx.commStream));
  }

  NCCL_CHECK_LOCAL(ncclGroupStart());
  for (int peer = 0; peer < ctx.size; ++peer) {
    if (peer == ctx.rank) {
      continue;
    }
    NCCL_CHECK_LOCAL(ncclSend(dSendPacked + sendOffsets[peer],
                              static_cast<size_t>(ctx.qCounts[peer]) * static_cast<size_t>(ctx.qLocalCols),
                              ncclDouble,
                              peer,
                              ctx.nccl,
                              ctx.commStream));
    NCCL_CHECK_LOCAL(ncclRecv(dRecvPacked + recvOffsets[peer],
                              static_cast<size_t>(ctx.qCounts[peer]) * static_cast<size_t>(ctx.qLocalCols),
                              ncclDouble,
                              peer,
                              ctx.nccl,
                              ctx.commStream));
  }
  NCCL_CHECK_LOCAL(ncclGroupEnd());

  const size_t selfElems = static_cast<size_t>(ctx.qLocalCols) * static_cast<size_t>(ctx.qLocalCols);
  EVD_CUDA_CHECK(cudaMemcpyAsync(dRecvPacked + recvOffsets[ctx.rank],
                        dSendPacked + sendOffsets[ctx.rank],
                        selfElems * sizeof(double),
                        cudaMemcpyDeviceToDevice,
                        ctx.commStream));
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));

  for (int src = 0; src < ctx.size; ++src) {
    const long srcCols = ctx.qCounts[src];
    dim3 block(16, 16);
    dim3 grid(ceilDiv(ctx.qLocalCols, block.x), ceilDiv(srcCols, block.y));
    unpackTransposedPeerBlock<<<grid, block>>>(dRecvPacked + recvOffsets[src],
                                               srcCols,
                                               dOutput,
                                               ldOutput,
                                               ctx.qDispls[src],
                                               ctx.qLocalCols);
    EVD_CUDA_CHECK(cudaGetLastError());
  }
  if (ldOutput > ctx.n) {
    dim3 clearBlock(32, 32);
    dim3 clearGrid(ceilDiv(ldOutput - ctx.n, clearBlock.x), ceilDiv(ctx.qLocalCols, clearBlock.y));
    launchClearMatrix(clearGrid,
                             clearBlock,
                             ldOutput - ctx.n,
                             ctx.qLocalCols,
                             dOutput + ctx.n,
                             ldOutput);
    EVD_CUDA_CHECK(cudaGetLastError());
  }
  EVD_CUDA_CHECK(cudaDeviceSynchronize());
}

