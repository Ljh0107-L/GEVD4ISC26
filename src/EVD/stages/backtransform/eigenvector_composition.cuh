double runFinalGemmStreaming(const DistContext &ctx,
                             cublasHandle_t cublas,
                             double *dQ,
                             long ldQ,
                             const double *hZ,
                             long tileCols,
                             bool useStartBarrier,
                             bool keepFinalQ,
                             double **dQFinalOut)
{
  const double one = 1.0;
  const double zero = 0.0;
  double *dZTile = nullptr;
  double *dQFinal = nullptr;
  double *dQFinalTile = nullptr;
  if (keepFinalQ) {
    EVD_CUDA_CHECK(evdMalloc(&dQFinal, sizeof(double) * static_cast<size_t>(ctx.qLocalCols) * ctx.n));
  } else {
    EVD_CUDA_CHECK(evdMalloc(&dQFinalTile, sizeof(double) * static_cast<size_t>(ctx.qLocalCols) * tileCols));
  }

  const bool overlapFinalGemm =
      envIntOrDefault("EVD_FINAL_GEMM_OVERLAP", 1) != 0 && ctx.n > tileCols;
  if (!overlapFinalGemm) {
    EVD_CUDA_CHECK(evdMalloc(&dZTile, sizeof(double) * static_cast<size_t>(ctx.n) * tileCols));
    if (useStartBarrier) {
      MPI_CHECK(MPI_Barrier(ctx.comm));
    }
    const double t0 = MPI_Wtime();
    for (long col = 0; col < ctx.n; col += tileCols) {
      const long curTile = std::min(tileCols, ctx.n - col);
      if (ctx.rank == 0) {
        EVD_CUDA_CHECK(cudaMemcpyAsync(dZTile,
                              hZ + static_cast<size_t>(col) * ctx.n,
                              sizeof(double) * static_cast<size_t>(ctx.n) * curTile,
                              cudaMemcpyHostToDevice,
                              ctx.commStream));
      }
      NCCL_CHECK_LOCAL(ncclBroadcast(dZTile,
                                     dZTile,
                                     static_cast<size_t>(ctx.n) * static_cast<size_t>(curTile),
                                     ncclDouble,
                                     0,
                                     ctx.nccl,
                                     ctx.commStream));
      EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
      CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     static_cast<int>(ctx.qLocalCols),
                                     static_cast<int>(curTile),
                                     static_cast<int>(ctx.n),
                                     &one,
                                     dQ,
                                     static_cast<int>(ldQ),
                                     dZTile,
                                     static_cast<int>(ctx.n),
                                     &zero,
                                     keepFinalQ ? (dQFinal + static_cast<size_t>(col) * ctx.qLocalCols) : dQFinalTile,
                                     static_cast<int>(ctx.qLocalCols)));
    }
    EVD_CUDA_CHECK(cudaDeviceSynchronize());
    const double t1 = MPI_Wtime();

    if (keepFinalQ) {
      *dQFinalOut = dQFinal;
    } else {
      if (dQFinalOut != nullptr) {
        *dQFinalOut = nullptr;
      }
      EVD_CUDA_CHECK(evdFree(dQFinalTile));
    }
    EVD_CUDA_CHECK(evdFree(dZTile));
    return (t1 - t0) * 1000.0;
  }

  double *dZTiles[2] = {nullptr, nullptr};
  EVD_CUDA_CHECK(evdMalloc(&dZTiles[0], sizeof(double) * static_cast<size_t>(ctx.n) * tileCols));
  EVD_CUDA_CHECK(evdMalloc(&dZTiles[1], sizeof(double) * static_cast<size_t>(ctx.n) * tileCols));
  cudaStream_t computeStream = nullptr;
  EVD_CUDA_CHECK(cudaStreamCreate(&computeStream));
  cudaEvent_t tileReady[2] = {nullptr, nullptr};
  cudaEvent_t gemmDone[2] = {nullptr, nullptr};
  for (int i = 0; i < 2; ++i) {
    EVD_CUDA_CHECK(cudaEventCreateWithFlags(&tileReady[i], cudaEventDisableTiming));
    EVD_CUDA_CHECK(cudaEventCreateWithFlags(&gemmDone[i], cudaEventDisableTiming));
  }
  cudaStream_t originalCublasStream = nullptr;
  CUBLAS_CHECK_LOCAL(cublasGetStream(cublas, &originalCublasStream));
  CUBLAS_CHECK_LOCAL(cublasSetStream(cublas, computeStream));
  bool gemmIssued[2] = {false, false};

  if (useStartBarrier) {
    MPI_CHECK(MPI_Barrier(ctx.comm));
  }
  const double t0 = MPI_Wtime();
  for (long col = 0; col < ctx.n; col += tileCols) {
    const long curTile = std::min(tileCols, ctx.n - col);
    const int buf = static_cast<int>((col / tileCols) % 2);
    if (gemmIssued[buf]) {
      EVD_CUDA_CHECK(cudaStreamWaitEvent(ctx.commStream, gemmDone[buf], 0));
    }
    if (ctx.rank == 0) {
      EVD_CUDA_CHECK(cudaMemcpyAsync(dZTiles[buf],
                            hZ + static_cast<size_t>(col) * ctx.n,
                            sizeof(double) * static_cast<size_t>(ctx.n) * curTile,
                            cudaMemcpyHostToDevice,
                            ctx.commStream));
    }
    NCCL_CHECK_LOCAL(ncclBroadcast(dZTiles[buf],
                                   dZTiles[buf],
                                   static_cast<size_t>(ctx.n) * static_cast<size_t>(curTile),
                                   ncclDouble,
                                   0,
                                   ctx.nccl,
                                   ctx.commStream));
    EVD_CUDA_CHECK(cudaEventRecord(tileReady[buf], ctx.commStream));
    EVD_CUDA_CHECK(cudaStreamWaitEvent(computeStream, tileReady[buf], 0));
    CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_N,
                                   static_cast<int>(ctx.qLocalCols),
                                   static_cast<int>(curTile),
                                   static_cast<int>(ctx.n),
                                   &one,
                                   dQ,
                                   static_cast<int>(ldQ),
                                   dZTiles[buf],
                                   static_cast<int>(ctx.n),
                                   &zero,
                                   keepFinalQ ? (dQFinal + static_cast<size_t>(col) * ctx.qLocalCols) : dQFinalTile,
                                   static_cast<int>(ctx.qLocalCols)));
    EVD_CUDA_CHECK(cudaEventRecord(gemmDone[buf], computeStream));
    gemmIssued[buf] = true;
  }
  EVD_CUDA_CHECK(cudaStreamSynchronize(computeStream));
  const double t1 = MPI_Wtime();
  CUBLAS_CHECK_LOCAL(cublasSetStream(cublas, originalCublasStream));

  if (keepFinalQ) {
    *dQFinalOut = dQFinal;
  } else {
    if (dQFinalOut != nullptr) {
      *dQFinalOut = nullptr;
    }
    EVD_CUDA_CHECK(evdFree(dQFinalTile));
  }
  for (int i = 0; i < 2; ++i) {
    EVD_CUDA_CHECK(cudaEventDestroy(tileReady[i]));
    EVD_CUDA_CHECK(cudaEventDestroy(gemmDone[i]));
    EVD_CUDA_CHECK(evdFree(dZTiles[i]));
  }
  EVD_CUDA_CHECK(cudaStreamDestroy(computeStream));
  return (t1 - t0) * 1000.0;
}

