bool debugStageProgressEnabled()
{
  return envIntOrDefault("EVD_DEBUG_STAGE_PROGRESS", 0) != 0;
}

bool printStageStatisticsEnabled()
{
  return envIntOrDefault("EVD_PRINT_STAGE_STATS", 0) != 0;
}

void allocateSbrWorkspace(const DistContext &ctx, SbrWorkspace *ws, bool allocateDoubleBlockBuffers)
{
  const long n = ctx.n;
  const long b = ctx.b;
  const long maxLocalCols = *std::max_element(ctx.counts.begin(), ctx.counts.end());
  EVD_CUDA_CHECK(evdMalloc(&ws->dWork, sizeof(double) * static_cast<size_t>(n) * ctx.nb));
  EVD_CUDA_CHECK(evdMalloc(&ws->dR, sizeof(double) * static_cast<size_t>(n) * b));
  EVD_CUDA_CHECK(evdMalloc(&ws->dWPanel, sizeof(double) * static_cast<size_t>(n) * b));
  EVD_CUDA_CHECK(evdMalloc(&ws->dYPanel, sizeof(double) * static_cast<size_t>(n) * b));
  EVD_CUDA_CHECK(evdMalloc(&ws->dWBT, sizeof(double) * static_cast<size_t>(b) * maxLocalCols));
  EVD_CUDA_CHECK(evdMalloc(&ws->dYBT, sizeof(double) * static_cast<size_t>(b) * maxLocalCols));
  EVD_CUDA_CHECK(evdMalloc(&ws->dAWBT, sizeof(double) * static_cast<size_t>(b) * maxLocalCols));
  EVD_CUDA_CHECK(evdMalloc(&ws->dZBT, sizeof(double) * static_cast<size_t>(b) * maxLocalCols));
  EVD_CUDA_CHECK(evdMalloc(&ws->dZFullBT, sizeof(double) * static_cast<size_t>(b) * n));
  EVD_CUDA_CHECK(evdMalloc(&ws->dSLocal, sizeof(double) * b * b));
  EVD_CUDA_CHECK(evdMalloc(&ws->dS, sizeof(double) * b * b));
  if (allocateDoubleBlockBuffers) {
    EVD_CUDA_CHECK(evdMalloc(&ws->dABase, sizeof(double) * static_cast<size_t>(n) * ctx.localCols));
    EVD_CUDA_CHECK(evdMalloc(&ws->dYBlock, sizeof(double) * static_cast<size_t>(n) * ctx.nb));
    EVD_CUDA_CHECK(evdMalloc(&ws->dYBlockBT, sizeof(double) * static_cast<size_t>(ctx.nb) * n));
    EVD_CUDA_CHECK(evdMalloc(&ws->dZBlockBT, sizeof(double) * static_cast<size_t>(ctx.nb) * n));
    EVD_CUDA_CHECK(evdMalloc(&ws->dTmpPrev1, sizeof(double) * static_cast<size_t>(ctx.nb) * b));
    EVD_CUDA_CHECK(evdMalloc(&ws->dTmpPrev2, sizeof(double) * static_cast<size_t>(ctx.nb) * b));
  }

}

void freeSbrWorkspace(SbrWorkspace *ws)
{
  if (ws->dWork) EVD_CUDA_CHECK(evdFree(ws->dWork));
  if (ws->dR) EVD_CUDA_CHECK(evdFree(ws->dR));
  if (ws->dWPanel) EVD_CUDA_CHECK(evdFree(ws->dWPanel));
  if (ws->dYPanel) EVD_CUDA_CHECK(evdFree(ws->dYPanel));
  if (ws->dWBT) EVD_CUDA_CHECK(evdFree(ws->dWBT));
  if (ws->dYBT) EVD_CUDA_CHECK(evdFree(ws->dYBT));
  if (ws->dAWBT) EVD_CUDA_CHECK(evdFree(ws->dAWBT));
  if (ws->dZBT) EVD_CUDA_CHECK(evdFree(ws->dZBT));
  if (ws->dZFullBT) EVD_CUDA_CHECK(evdFree(ws->dZFullBT));
  if (ws->dSLocal) EVD_CUDA_CHECK(evdFree(ws->dSLocal));
  if (ws->dS) EVD_CUDA_CHECK(evdFree(ws->dS));
  if (ws->dABase) EVD_CUDA_CHECK(evdFree(ws->dABase));
  if (ws->dYBlock) EVD_CUDA_CHECK(evdFree(ws->dYBlock));
  if (ws->dYBlockBT) EVD_CUDA_CHECK(evdFree(ws->dYBlockBT));
  if (ws->dZBlockBT) EVD_CUDA_CHECK(evdFree(ws->dZBlockBT));
  if (ws->dTmpPrev1) EVD_CUDA_CHECK(evdFree(ws->dTmpPrev1));
  if (ws->dTmpPrev2) EVD_CUDA_CHECK(evdFree(ws->dTmpPrev2));
}

void initSbrSuffixNccls(DistContext *ctx)
{
  ctx->sbrSuffixNccls.assign(ctx->size, nullptr);
  // The zero suffix is exactly the already-created full communicator.
  // Reusing it avoids a redundant cross-node NCCL communicator setup.
  if (ctx->size > 0) {
    ctx->sbrSuffixNccls[0] = ctx->nccl;
  }
  for (int firstRank = 1; firstRank < ctx->size; ++firstRank) {
    ncclUniqueId suffixId;
    if (ctx->rank == firstRank) {
      NCCL_CHECK_LOCAL(ncclGetUniqueId(&suffixId));
    }
    MPI_CHECK(MPI_Bcast(&suffixId, sizeof(suffixId), MPI_BYTE, firstRank, ctx->comm));
    if (ctx->rank >= firstRank) {
      [[maybe_unused]]
      gevd4isc26::detail::ScopedCpuAffinityRestore preserve_cpu_affinity;
      NCCL_CHECK_LOCAL(ncclCommInitRank(&ctx->sbrSuffixNccls[firstRank],
                                        ctx->size - firstRank,
                                        suffixId,
                                        ctx->rank - firstRank));
    }
  }
}

void destroySbrSuffixNccls(DistContext *ctx)
{
  for (ncclComm_t comm : ctx->sbrSuffixNccls) {
    if (comm != nullptr && comm != ctx->nccl) {
      NCCL_CHECK_LOCAL(ncclCommDestroy(comm));
    }
  }
  ctx->sbrSuffixNccls.clear();
}

void broadcastWYPanelsDevice(const DistContext &ctx,
                             ncclComm_t nccl,
                             double *dLocalWPanel,
                             double *dLocalYPanel,
                             long ldLocalPanel,
                             double *dWPanel,
                             double *dYPanel,
                             long rows,
                             long cols,
                             int owner,
                             int rootInComm)
{
  if (ctx.rank == owner) {
    EVD_CUDA_CHECK(cudaMemcpy2DAsync(dWPanel,
                            rows * sizeof(double),
                            dLocalWPanel,
                            ldLocalPanel * sizeof(double),
                            rows * sizeof(double),
                            cols,
                            cudaMemcpyDeviceToDevice,
                            ctx.commStream));
    EVD_CUDA_CHECK(cudaMemcpy2DAsync(dYPanel,
                            rows * sizeof(double),
                            dLocalYPanel,
                            ldLocalPanel * sizeof(double),
                            rows * sizeof(double),
                            cols,
                            cudaMemcpyDeviceToDevice,
                            ctx.commStream));
  }

  const size_t panelEntries = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  NCCL_CHECK_LOCAL(ncclGroupStart());
  NCCL_CHECK_LOCAL(ncclBroadcast(dWPanel,
                                 dWPanel,
                                 panelEntries,
                                 ncclDouble,
                                 rootInComm,
                                 nccl,
                                 ctx.commStream));
  NCCL_CHECK_LOCAL(ncclBroadcast(dYPanel,
                                 dYPanel,
                                 panelEntries,
                                 ncclDouble,
                                 rootInComm,
                                 nccl,
                                 ctx.commStream));
  NCCL_CHECK_LOCAL(ncclGroupEnd());
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
}

void broadcastZActiveSlices(const DistContext &ctx,
                            ncclComm_t nccl,
                            int firstRank,
                            double *dZCompactBT,
                            long activeCols,
                            long activeGlobalStart,
                            double *dZFullBT,
                            long panelRowStart)
{
  EVD_CUDA_CHECK(cudaDeviceSynchronize());
  if (activeCols > 0) {
    EVD_CUDA_CHECK(cudaMemcpyAsync(dZFullBT + activeGlobalStart * ctx.b,
                          dZCompactBT,
                          sizeof(double) * static_cast<size_t>(ctx.b) * static_cast<size_t>(activeCols),
                          cudaMemcpyDeviceToDevice,
                          ctx.commStream));
  }

  NCCL_CHECK_LOCAL(ncclGroupStart());
  for (int peer = firstRank; peer < ctx.size; ++peer) {
    const long peerStart = ctx.displs[peer];
    const long peerEnd = peerStart + ctx.counts[peer];
    const long sliceStart = std::max(panelRowStart, peerStart);
    if (peerEnd <= sliceStart) {
      continue;
    }
    const long sliceCols = peerEnd - sliceStart;
    double *slice = dZFullBT + sliceStart * ctx.b;
    NCCL_CHECK_LOCAL(ncclBroadcast(slice,
                                   slice,
                                   static_cast<size_t>(ctx.b) * static_cast<size_t>(sliceCols),
                                   ncclDouble,
                                   peer - firstRank,
                                   nccl,
                                   ctx.commStream));
  }
  NCCL_CHECK_LOCAL(ncclGroupEnd());
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
}
