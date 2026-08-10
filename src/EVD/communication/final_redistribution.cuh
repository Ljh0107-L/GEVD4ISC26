__global__ void unpackColumnMajorPeerBlock(const double *src,
                                           long srcRows,
                                           double *dst,
                                           long ldDst,
                                           long dstRowStart,
                                           long dstCols)
{
  const long row = blockIdx.x * blockDim.x + threadIdx.x;
  const long col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < srcRows && col < dstCols) {
    dst[dstRowStart + row + col * ldDst] = src[row + col * srcRows];
  }
}

void redistributeFinalRowsToColumns(const DistContext &ctx,
                                    const double *dRowBlock,
                                    double *dColumnBlock,
                                    long ldColumnBlock)
{
  const long localCols = ctx.localCols;
  std::vector<size_t> recvOffsets(ctx.size, 0);
  size_t offset = 0;
  for (int src = 0; src < ctx.size; ++src) {
    recvOffsets[src] = offset;
    offset += static_cast<size_t>(ctx.qCounts[src]) * static_cast<size_t>(localCols);
  }

  double *dPacked = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dPacked, sizeof(double) * static_cast<size_t>(ctx.n) *
                                 static_cast<size_t>(localCols)));

  NCCL_CHECK_LOCAL(ncclGroupStart());
  for (int peer = 0; peer < ctx.size; ++peer) {
    if (peer == ctx.rank) {
      continue;
    }
    NCCL_CHECK_LOCAL(ncclSend(dRowBlock + static_cast<size_t>(ctx.displs[peer]) *
                                            static_cast<size_t>(ctx.qLocalCols),
                              static_cast<size_t>(ctx.qLocalCols) *
                                  static_cast<size_t>(ctx.counts[peer]),
                              ncclDouble,
                              peer,
                              ctx.nccl,
                              ctx.commStream));
    NCCL_CHECK_LOCAL(ncclRecv(dPacked + recvOffsets[peer],
                              static_cast<size_t>(ctx.qCounts[peer]) *
                                  static_cast<size_t>(localCols),
                              ncclDouble,
                              peer,
                              ctx.nccl,
                              ctx.commStream));
  }
  NCCL_CHECK_LOCAL(ncclGroupEnd());

  EVD_CUDA_CHECK(cudaMemcpyAsync(dPacked + recvOffsets[ctx.rank],
                        dRowBlock + static_cast<size_t>(ctx.colStart) *
                                      static_cast<size_t>(ctx.qLocalCols),
                        sizeof(double) * static_cast<size_t>(ctx.qLocalCols) *
                            static_cast<size_t>(localCols),
                        cudaMemcpyDeviceToDevice,
                        ctx.commStream));
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));

  for (int src = 0; src < ctx.size; ++src) {
    dim3 block(16, 16);
    dim3 grid(ceilDiv(ctx.qCounts[src], block.x), ceilDiv(localCols, block.y));
    unpackColumnMajorPeerBlock<<<grid, block>>>(dPacked + recvOffsets[src],
                                                ctx.qCounts[src],
                                                dColumnBlock,
                                                ldColumnBlock,
                                                ctx.qDispls[src],
                                                localCols);
    EVD_CUDA_CHECK(cudaGetLastError());
  }
  EVD_CUDA_CHECK(cudaDeviceSynchronize());
  EVD_CUDA_CHECK(evdFree(dPacked));
}
