void gatherBandToAll(const DistContext &ctx, double *dA, double *dBand)
{
  const long ldBand = 2 * ctx.b;
  if (allCountsEqual(ctx.counts)) {
    const size_t localElems = static_cast<size_t>(ldBand) * ctx.localCols;
    double *dLocalBand = nullptr;
    EVD_CUDA_CHECK(evdMalloc(&dLocalBand, localElems * sizeof(double)));
    dim3 block(32, 8);
    dim3 grid(ceilDiv(ldBand, block.x), ceilDiv(ctx.localCols, block.y));
    packBandColumns<<<grid, block>>>(dA, ctx.n, ctx.n, ctx.b, ctx.colStart, ctx.localCols, dLocalBand, ldBand);
    EVD_CUDA_CHECK(cudaGetLastError());

    NCCL_CHECK_LOCAL(ncclAllGather(dLocalBand,
                                   dBand,
                                   localElems,
                                   ncclDouble,
                                   ctx.nccl,
                                   ctx.commStream));
    EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
    EVD_CUDA_CHECK(evdFree(dLocalBand));
    return;
  }

  dim3 block(32, 8);
  dim3 grid(ceilDiv(ldBand, block.x), ceilDiv(ctx.localCols, block.y));
  packBandColumns<<<grid, block>>>(dA,
                                  ctx.n,
                                  ctx.n,
                                  ctx.b,
                                  ctx.colStart,
                                  ctx.localCols,
                                  dBand + ctx.colStart * ldBand,
                                  ldBand);
  EVD_CUDA_CHECK(cudaGetLastError());
  EVD_CUDA_CHECK(cudaDeviceSynchronize());

  NCCL_CHECK_LOCAL(ncclGroupStart());
  for (int root = 0; root < ctx.size; ++root) {
    const size_t elems = static_cast<size_t>(ldBand) * static_cast<size_t>(ctx.counts[root]);
    if (elems == 0) {
      continue;
    }
    double *segment = dBand + ctx.displs[root] * ldBand;
    NCCL_CHECK_LOCAL(ncclBroadcast(segment,
                                   segment,
                                   elems,
                                   ncclDouble,
                                   root,
                                   ctx.nccl,
                                   ctx.commStream));
  }
  NCCL_CHECK_LOCAL(ncclGroupEnd());
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
}

