void broadcastBandSuffix(const DistContext &ctx, double *dBand, long syncCol, int owner)
{
  const long ldBand = 2 * ctx.b;
  const long colStart = std::max<long>(0, std::min<long>(ctx.n, syncCol));
  if (colStart >= ctx.n) {
    return;
  }
  NCCL_CHECK_LOCAL(ncclBroadcast(dBand + colStart * ldBand,
                                 dBand + colStart * ldBand,
                                 static_cast<size_t>(ldBand) * static_cast<size_t>(ctx.n - colStart),
                                 ncclDouble,
                                 owner,
                                 ctx.nccl,
                                 ctx.commStream));
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
}

void broadcastBandColumnRange(const DistContext &ctx,
                              double *dBand,
                              long syncColStart,
                              long syncColEnd,
                              int owner)
{
  const long ldBand = 2 * ctx.b;
  const long colStart = std::max<long>(0, std::min<long>(ctx.n, syncColStart));
  const long colEnd = std::max<long>(colStart, std::min<long>(ctx.n, syncColEnd));
  if (colStart >= colEnd) {
    return;
  }
  NCCL_CHECK_LOCAL(ncclBroadcast(dBand + colStart * ldBand,
                                 dBand + colStart * ldBand,
                                 static_cast<size_t>(ldBand) * static_cast<size_t>(colEnd - colStart),
                                 ncclDouble,
                                 owner,
                                 ctx.nccl,
                                 ctx.commStream));
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
}

void gatherOwnedBandColumnsToAll(const DistContext &ctx, double *dBand)
{
  for (int owner = 0; owner < ctx.size; ++owner) {
    const long start = ctx.displs[owner];
    const long end = ctx.displs[owner] + ctx.counts[owner];
    broadcastBandColumnRange(ctx, dBand, start, end, owner);
  }
}

void broadcastBandColumnRangeOnComm(const DistContext &ctx,
                                    ncclComm_t comm,
                                    double *dBand,
                                    long syncColStart,
                                    long syncColEnd,
                                    int rootInComm)
{
  const long ldBand = 2 * ctx.b;
  const long colStart = std::max<long>(0, std::min<long>(ctx.n, syncColStart));
  const long colEnd = std::max<long>(colStart, std::min<long>(ctx.n, syncColEnd));
  if (colStart >= colEnd) {
    return;
  }
  double *range = dBand + colStart * ldBand;
  NCCL_CHECK_LOCAL(ncclBroadcast(range,
                                 range,
                                 static_cast<size_t>(ldBand) * static_cast<size_t>(colEnd - colStart),
                                 ncclDouble,
                                 rootInComm,
                                 comm,
                                 ctx.commStream));
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
}

void broadcastBandColumnRangeSuffix(const DistContext &ctx,
                                    double *dBand,
                                    long syncColStart,
                                    long syncColEnd,
                                    int owner)
{
  const int firstRank = std::max(0, std::min(owner, ctx.size - 1));
  if (firstRank == 0) {
    broadcastBandColumnRange(ctx, dBand, syncColStart, syncColEnd, owner);
    return;
  }
  if (ctx.rank < firstRank) {
    return;
  }
  if (ctx.sbrSuffixNccls.empty() || ctx.sbrSuffixNccls[firstRank] == nullptr) {
    if (ctx.rank == firstRank) {
      std::cerr << "BC suffix band broadcast requires initialized suffix NCCL comms; "
                << "enable EVD_BC_SEGMENTED_TILE_SUFFIX_BCAST through the main path"
                << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  broadcastBandColumnRangeOnComm(ctx,
                                 ctx.sbrSuffixNccls[firstRank],
                                 dBand,
                                 syncColStart,
                                 syncColEnd,
                                 owner - firstRank);
}

void packAndBroadcastCompletedBandBlock(const DistContext &ctx,
                                        double *dA,
                                        double *dBand,
                                        long blockStart,
                                        long blockEnd,
                                        int commFirstRank)
{
  const long ldBand = 2 * ctx.b;
  const long localStart = std::max(blockStart, ctx.colStart);
  const long localEnd = std::min(blockEnd, ctx.colStart + ctx.localCols);
  if (localEnd > localStart) {
    const long localOffset = localStart - ctx.colStart;
    const long cols = localEnd - localStart;
    dim3 block(32, 8);
    dim3 grid(ceilDiv(ldBand, block.x), ceilDiv(cols, block.y));
    packBandColumns<<<grid, block>>>(dA + localOffset * ctx.n,
                                    ctx.n,
                                    ctx.n,
                                    ctx.b,
                                    localStart,
                                    cols,
                                    dBand + localStart * ldBand,
                                    ldBand);
    EVD_CUDA_CHECK(cudaGetLastError());
  }
  EVD_CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long blockDoubles = 0;
  const int firstRank = std::max(0, std::min(commFirstRank, ctx.size - 1));
  const bool rankInBandComm = ctx.rank >= firstRank;
  ncclComm_t bandComm = (firstRank == 0) ? ctx.nccl : ctx.sbrSuffixNccls[firstRank];
  for (int owner = 0; owner < ctx.size; ++owner) {
    const long ownerStart = std::max(blockStart, ctx.displs[owner]);
    const long ownerEnd = std::min(blockEnd, ctx.displs[owner] + ctx.counts[owner]);
    if (ownerEnd <= ownerStart) {
      continue;
    }
    if (owner < firstRank) {
      continue;
    }
    blockDoubles += static_cast<unsigned long long>(ldBand) *
                    static_cast<unsigned long long>(ownerEnd - ownerStart);
    if (rankInBandComm) {
      broadcastBandColumnRangeOnComm(ctx,
                                     bandComm,
                                     dBand,
                                     ownerStart,
                                     ownerEnd,
                                     owner - firstRank);
    }
  }

  if (ctx.rank == 0 && envIntOrDefault("EVD_DEBUG_STAGE_PROGRESS", 0) != 0) {
    std::cout << "SBR band-pipeline block [" << blockStart << "," << blockEnd
              << ") first_rank=" << firstRank
              << " broadcast="
              << static_cast<double>(blockDoubles * sizeof(double)) / (1024.0 * 1024.0)
              << " MiB" << std::endl;
  }
}

void handoffBandRangeToNextOwner(const DistContext &ctx,
                                 double *dBand,
                                 long syncColStart,
                                 long syncColEnd,
                                 int owner)
{
  if (owner + 1 >= ctx.size) {
    return;
  }
  const long ldBand = 2 * ctx.b;
  const long colStart = std::max<long>(0, std::min<long>(ctx.n, syncColStart));
  const long colEnd = std::max<long>(colStart, std::min<long>(ctx.n, syncColEnd));
  if (colStart >= colEnd) {
    return;
  }
  double *range = dBand + colStart * ldBand;
  const size_t elems = static_cast<size_t>(ldBand) * static_cast<size_t>(colEnd - colStart);
  if (ctx.rank == owner || ctx.rank == owner + 1) {
    NCCL_CHECK_LOCAL(ncclGroupStart());
    if (ctx.rank == owner) {
      NCCL_CHECK_LOCAL(ncclSend(range,
                                elems,
                                ncclDouble,
                                owner + 1,
                                ctx.nccl,
                                ctx.commStream));
    } else {
      NCCL_CHECK_LOCAL(ncclRecv(range,
                                elems,
                                ncclDouble,
                                owner,
                                ctx.nccl,
                                ctx.commStream));
    }
    NCCL_CHECK_LOCAL(ncclGroupEnd());
    EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
  }
}

void handoffBandRangeToPeer(const DistContext &ctx,
                            double *dBand,
                            long syncColStart,
                            long syncColEnd,
                            int srcOwner,
                            int dstOwner)
{
  if (srcOwner == dstOwner) {
    return;
  }
  if (srcOwner < 0 || srcOwner >= ctx.size || dstOwner < 0 || dstOwner >= ctx.size) {
    if (ctx.rank == 0) {
      std::cerr << "Invalid BC band handoff peer: src=" << srcOwner
                << " dst=" << dstOwner << " size=" << ctx.size << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  const long ldBand = 2 * ctx.b;
  const long colStart = std::max<long>(0, std::min<long>(ctx.n, syncColStart));
  const long colEnd = std::max<long>(colStart, std::min<long>(ctx.n, syncColEnd));
  if (colStart >= colEnd) {
    return;
  }
  if (debugStageProgressEnabled() && (ctx.rank == srcOwner || ctx.rank == dstOwner)) {
    std::cout << "StageProgress rank=" << ctx.rank
              << " phase=band_handoff src=" << srcOwner
              << " dst=" << dstOwner
              << " cols=[" << colStart << "," << colEnd << ")"
              << " doubles=" << (2 * ctx.b * (colEnd - colStart))
              << std::endl;
  }
  double *range = dBand + colStart * ldBand;
  const size_t elems = static_cast<size_t>(ldBand) * static_cast<size_t>(colEnd - colStart);
  if (ctx.rank == srcOwner || ctx.rank == dstOwner) {
    NCCL_CHECK_LOCAL(ncclGroupStart());
    if (ctx.rank == srcOwner) {
      NCCL_CHECK_LOCAL(ncclSend(range,
                                elems,
                                ncclDouble,
                                dstOwner,
                                ctx.nccl,
                                ctx.commStream));
    } else {
      NCCL_CHECK_LOCAL(ncclRecv(range,
                                elems,
                                ncclDouble,
                                srcOwner,
                                ctx.nccl,
                                ctx.commStream));
    }
    NCCL_CHECK_LOCAL(ncclGroupEnd());
    EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
  }
}

int latestRowTileSourceOwnerForColumn(const DistContext &ctx,
                                      long col,
                                      long tileRows,
                                      long tileCount,
                                      long haloCols)
{
  int latestTile = -1;
  for (long tile = 0; tile < tileCount; ++tile) {
    const long rowTileStart = tile * tileRows;
    const long rowTileEnd = std::min<long>(ctx.n, rowTileStart + tileRows);
    const long bandStart = std::max<long>(0, rowTileStart - haloCols);
    const long bandEnd = std::min<long>(ctx.n, rowTileEnd + haloCols);
    if (col >= bandStart && col < bandEnd) {
      latestTile = static_cast<int>(tile);
    }
  }
  if (latestTile < 0) {
    return ownerOfColumn(std::min<long>(ctx.n - 1, std::max<long>(0, col)),
                         ctx.counts,
                         ctx.displs);
  }
  const long sourceTileStart = std::min<long>(ctx.n - 1,
                                             static_cast<long>(latestTile) * tileRows);
  return ownerOfColumn(sourceTileStart, ctx.counts, ctx.displs);
}

int encodeSuffixBandAvailability(int firstRank)
{
  return -std::max(0, firstRank) - 1;
}

bool encodedBandAvailableOnOwner(int encodedOwner, int dstOwner)
{
  if (encodedOwner >= 0) {
    return encodedOwner == dstOwner;
  }
  const int firstRank = -encodedOwner - 1;
  return dstOwner >= firstRank;
}

int encodedBandSourceOwner(int encodedOwner)
{
  return encodedOwner >= 0 ? encodedOwner : (-encodedOwner - 1);
}

unsigned long long handoffRowTileWrapRangeToOwner(const DistContext &ctx,
                                                  double *dBand,
                                                  long syncColStart,
                                                  long syncColEnd,
                                                  long tileRows,
                                                  long tileCount,
                                                  long haloCols,
                                                  int dstOwner,
                                                  long *maxCols)
{
  const long colStart = std::max<long>(0, std::min<long>(ctx.n, syncColStart));
  const long colEnd = std::max<long>(colStart, std::min<long>(ctx.n, syncColEnd));
  if (colStart >= colEnd) {
    return 0;
  }

  unsigned long long sentDoubles = 0;
  long segmentStart = colStart;
  int segmentOwner =
      latestRowTileSourceOwnerForColumn(ctx, segmentStart, tileRows, tileCount, haloCols);
  for (long col = colStart + 1; col <= colEnd; ++col) {
    const int owner = (col < colEnd)
                          ? latestRowTileSourceOwnerForColumn(ctx,
                                                              col,
                                                              tileRows,
                                                              tileCount,
                                                              haloCols)
                          : -1;
    if (col < colEnd && owner == segmentOwner) {
      continue;
    }
    const long cols = col - segmentStart;
    if (cols > 0) {
      if (maxCols != nullptr) {
        *maxCols = std::max<long>(*maxCols, cols);
      }
      if (!encodedBandAvailableOnOwner(segmentOwner, dstOwner)) {
        sentDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                       static_cast<unsigned long long>(cols);
        handoffBandRangeToPeer(ctx,
                               dBand,
                               segmentStart,
                               col,
                               encodedBandSourceOwner(segmentOwner),
                               dstOwner);
      }
    }
    segmentStart = col;
    segmentOwner = owner;
  }
  return sentDoubles;
}

unsigned long long pullBandRangeToOwner(const DistContext &ctx,
                                        double *dBand,
                                        long syncColStart,
                                        long syncColEnd,
                                        int dstOwner,
                                        const std::vector<int> &latestOwners,
                                        long *maxCols,
                                        unsigned long long *pullCallCount = nullptr,
                                        unsigned long long *pullTransferCount = nullptr,
                                        unsigned long long *maxPullTransfersPerCall = nullptr)
{
  const long colStart = std::max<long>(0, std::min<long>(ctx.n, syncColStart));
  const long colEnd = std::max<long>(colStart, std::min<long>(ctx.n, syncColEnd));
  if (colStart >= colEnd) {
    return 0;
  }
  if (pullCallCount != nullptr) {
    (*pullCallCount)++;
  }

  struct BandPullTransfer {
    long colStart = 0;
    long colEnd = 0;
    int srcOwner = -1;
  };

  const long ldBand = 2 * ctx.b;
  const bool groupedPull =
      envIntOrDefault("EVD_BC_SEGMENTED_PULL_GROUPED", 0) != 0;
  std::vector<BandPullTransfer> transfers;
  unsigned long long sentDoubles = 0;
  unsigned long long localTransferCount = 0;
  long segmentStart = colStart;
  int segmentOwner = latestOwners[static_cast<size_t>(segmentStart)];
  for (long col = colStart + 1; col <= colEnd; ++col) {
    const int owner = (col < colEnd) ? latestOwners[static_cast<size_t>(col)] : -1;
    if (col < colEnd && owner == segmentOwner) {
      continue;
    }
    const long cols = col - segmentStart;
    if (cols > 0) {
      if (maxCols != nullptr) {
        *maxCols = std::max<long>(*maxCols, cols);
      }
      if (!encodedBandAvailableOnOwner(segmentOwner, dstOwner)) {
        sentDoubles += static_cast<unsigned long long>(2 * ctx.b) *
                       static_cast<unsigned long long>(cols);
        localTransferCount++;
        const int srcOwner = encodedBandSourceOwner(segmentOwner);
        if (groupedPull) {
          transfers.push_back({segmentStart, col, srcOwner});
        } else {
          handoffBandRangeToPeer(ctx,
                                 dBand,
                                 segmentStart,
                                 col,
                                 srcOwner,
                                 dstOwner);
        }
      }
    }
    segmentStart = col;
    segmentOwner = owner;
  }

  if (groupedPull && !transfers.empty()) {
    bool participates = (ctx.rank == dstOwner);
    for (const BandPullTransfer &transfer : transfers) {
      participates = participates || (ctx.rank == transfer.srcOwner);
    }
    if (participates) {
      NCCL_CHECK_LOCAL(ncclGroupStart());
      for (const BandPullTransfer &transfer : transfers) {
        if (debugStageProgressEnabled() &&
            (ctx.rank == transfer.srcOwner || ctx.rank == dstOwner)) {
          std::cout << "StageProgress rank=" << ctx.rank
                    << " phase=band_pull_handoff src=" << transfer.srcOwner
                    << " dst=" << dstOwner
                    << " cols=[" << transfer.colStart << "," << transfer.colEnd << ")"
                    << " doubles=" << (ldBand * (transfer.colEnd - transfer.colStart))
                    << std::endl;
        }
        double *range = dBand + transfer.colStart * ldBand;
        const size_t elems = static_cast<size_t>(ldBand) *
                             static_cast<size_t>(transfer.colEnd - transfer.colStart);
        if (ctx.rank == transfer.srcOwner) {
          NCCL_CHECK_LOCAL(ncclSend(range,
                                    elems,
                                    ncclDouble,
                                    dstOwner,
                                    ctx.nccl,
                                    ctx.commStream));
        }
        if (ctx.rank == dstOwner) {
          NCCL_CHECK_LOCAL(ncclRecv(range,
                                    elems,
                                    ncclDouble,
                                    transfer.srcOwner,
                                    ctx.nccl,
                                    ctx.commStream));
        }
      }
      NCCL_CHECK_LOCAL(ncclGroupEnd());
      EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
    }
  }
  if (pullTransferCount != nullptr) {
    *pullTransferCount += localTransferCount;
  }
  if (maxPullTransfersPerCall != nullptr) {
    *maxPullTransfersPerCall =
        std::max(*maxPullTransfersPerCall, localTransferCount);
  }
  return sentDoubles;
}

void markBandRangeOwner(long n,
                        long syncColStart,
                        long syncColEnd,
                        int owner,
                        std::vector<int> *latestOwners)
{
  const long colStart = std::max<long>(0, std::min<long>(n, syncColStart));
  const long colEnd = std::max<long>(colStart, std::min<long>(n, syncColEnd));
  for (long col = colStart; col < colEnd; ++col) {
    (*latestOwners)[static_cast<size_t>(col)] = owner;
  }
}

void handoffBandRangeToPrevOwner(const DistContext &ctx,
                                 double *dBand,
                                 long syncColStart,
                                 long syncColEnd,
                                 int owner)
{
  if (owner <= 0) {
    return;
  }
  const long ldBand = 2 * ctx.b;
  const long colStart = std::max<long>(0, std::min<long>(ctx.n, syncColStart));
  const long colEnd = std::max<long>(colStart, std::min<long>(ctx.n, syncColEnd));
  if (colStart >= colEnd) {
    return;
  }
  double *range = dBand + colStart * ldBand;
  const size_t elems = static_cast<size_t>(ldBand) * static_cast<size_t>(colEnd - colStart);
  if (ctx.rank == owner || ctx.rank == owner - 1) {
    NCCL_CHECK_LOCAL(ncclGroupStart());
    if (ctx.rank == owner) {
      NCCL_CHECK_LOCAL(ncclSend(range,
                                elems,
                                ncclDouble,
                                owner - 1,
                                ctx.nccl,
                                ctx.commStream));
    } else {
      NCCL_CHECK_LOCAL(ncclRecv(range,
                                elems,
                                ncclDouble,
                                owner,
                                ctx.nccl,
                                ctx.commStream));
    }
    NCCL_CHECK_LOCAL(ncclGroupEnd());
    EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
  }
}

void handoffBandRangeToOwnerPrefixByNeighborChain(const DistContext &ctx,
                                                  double *dBand,
                                                  long syncColStart,
                                                  long syncColEnd,
                                                  int owner)
{
  const long colStart = std::max<long>(0, std::min<long>(ctx.n, syncColStart));
  const long colEnd = std::max<long>(colStart, std::min<long>(ctx.n, syncColEnd));
  if (colStart >= colEnd) {
    return;
  }
  for (int src = owner; src > 0; --src) {
    handoffBandRangeToPeer(ctx, dBand, colStart, colEnd, src, src - 1);
  }
}

void handoffBandRangeToOwnerSuffixByNeighborChain(const DistContext &ctx,
                                                  double *dBand,
                                                  long syncColStart,
                                                  long syncColEnd,
                                                  int owner)
{
  const long colStart = std::max<long>(0, std::min<long>(ctx.n, syncColStart));
  const long colEnd = std::max<long>(colStart, std::min<long>(ctx.n, syncColEnd));
  if (colStart >= colEnd) {
    return;
  }
  for (int src = owner; src + 1 < ctx.size; ++src) {
    handoffBandRangeToPeer(ctx, dBand, colStart, colEnd, src, src + 1);
  }
}

void handoffDeviceDoublesToNextOwner(const DistContext &ctx,
                                     double *buffer,
                                     long elems,
                                     int owner)
{
  if (owner + 1 >= ctx.size || elems <= 0) {
    return;
  }
  if (ctx.rank == owner || ctx.rank == owner + 1) {
    NCCL_CHECK_LOCAL(ncclGroupStart());
    if (ctx.rank == owner) {
      NCCL_CHECK_LOCAL(ncclSend(buffer,
                                static_cast<size_t>(elems),
                                ncclDouble,
                                owner + 1,
                                ctx.nccl,
                                ctx.commStream));
    } else {
      NCCL_CHECK_LOCAL(ncclRecv(buffer,
                                static_cast<size_t>(elems),
                                ncclDouble,
                                owner,
                                ctx.nccl,
                                ctx.commStream));
    }
    NCCL_CHECK_LOCAL(ncclGroupEnd());
    EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
  }
}

void handoffComProgressToNextOwner(const DistContext &ctx,
                                   std::vector<int> *hCom,
                                   int owner)
{
  if (owner + 1 >= ctx.size || hCom->empty()) {
    return;
  }
  constexpr int kComProgressTagBase = 2100;
  if (ctx.rank == owner) {
    MPI_CHECK(MPI_Send(hCom->data(),
                       static_cast<int>(hCom->size()),
                       MPI_INT,
                       owner + 1,
                       kComProgressTagBase + owner,
                       ctx.comm));
  } else if (ctx.rank == owner + 1) {
    MPI_CHECK(MPI_Recv(hCom->data(),
                       static_cast<int>(hCom->size()),
                       MPI_INT,
                       owner,
                       kComProgressTagBase + owner,
                       ctx.comm,
                       MPI_STATUS_IGNORE));
  }
}

void handoffComProgressToPeer(const DistContext &ctx,
                              std::vector<int> *hCom,
                              int srcOwner,
                              int dstOwner)
{
  if (srcOwner == dstOwner || hCom->empty()) {
    return;
  }
  if (srcOwner < 0 || srcOwner >= ctx.size || dstOwner < 0 || dstOwner >= ctx.size) {
    if (ctx.rank == 0) {
      std::cerr << "Invalid BC com handoff peer: src=" << srcOwner
                << " dst=" << dstOwner << " size=" << ctx.size << std::endl;
    }
    MPI_Abort(ctx.comm, 2);
  }
  constexpr int kComProgressPeerTagBase = 3200;
  const int tag = kComProgressPeerTagBase + srcOwner * ctx.size + dstOwner;
  if (ctx.rank == srcOwner) {
    MPI_CHECK(MPI_Send(hCom->data(),
                       static_cast<int>(hCom->size()),
                       MPI_INT,
                       dstOwner,
                       tag,
                       ctx.comm));
  } else if (ctx.rank == dstOwner) {
    MPI_CHECK(MPI_Recv(hCom->data(),
                       static_cast<int>(hCom->size()),
                       MPI_INT,
                       srcOwner,
                       tag,
                       ctx.comm,
                       MPI_STATUS_IGNORE));
  }
}

void handoffComRangeToNextOwner(const DistContext &ctx,
                                std::vector<int> *hCom,
                                long start,
                                long end,
                                int owner)
{
  if (owner + 1 >= ctx.size || hCom->empty()) {
    return;
  }
  const long rangeStart = std::max<long>(0, std::min<long>(static_cast<long>(hCom->size()), start));
  const long rangeEnd = std::max<long>(rangeStart,
                                       std::min<long>(static_cast<long>(hCom->size()), end));
  if (rangeEnd <= rangeStart) {
    return;
  }
  if (debugStageProgressEnabled() && (ctx.rank == owner || ctx.rank == owner + 1)) {
    std::cout << "StageProgress rank=" << ctx.rank
              << " phase=com_handoff_next owner=" << owner
              << " dst=" << (owner + 1)
              << " sweeps=[" << rangeStart << "," << rangeEnd << ")"
              << std::endl;
  }
  constexpr int kComRangeTagBase = 2600;
  int *range = hCom->data() + rangeStart;
  const int count = static_cast<int>(rangeEnd - rangeStart);
  if (ctx.rank == owner) {
    MPI_CHECK(MPI_Send(range,
                       count,
                       MPI_INT,
                       owner + 1,
                       kComRangeTagBase + owner,
                       ctx.comm));
  } else if (ctx.rank == owner + 1) {
    MPI_CHECK(MPI_Recv(range,
                       count,
                       MPI_INT,
                       owner,
                       kComRangeTagBase + owner,
                       ctx.comm,
                       MPI_STATUS_IGNORE));
  }
}

void handoffComRangeToPrevOwner(const DistContext &ctx,
                                std::vector<int> *hCom,
                                long start,
                                long end,
                                int owner)
{
  if (owner <= 0 || hCom->empty()) {
    return;
  }
  const long rangeStart = std::max<long>(0, std::min<long>(static_cast<long>(hCom->size()), start));
  const long rangeEnd = std::max<long>(rangeStart,
                                       std::min<long>(static_cast<long>(hCom->size()), end));
  if (rangeEnd <= rangeStart) {
    return;
  }
  if (debugStageProgressEnabled() && (ctx.rank == owner || ctx.rank == owner - 1)) {
    std::cout << "StageProgress rank=" << ctx.rank
              << " phase=com_handoff_prev owner=" << owner
              << " dst=" << (owner - 1)
              << " sweeps=[" << rangeStart << "," << rangeEnd << ")"
              << std::endl;
  }
  constexpr int kComRangeBackTagBase = 2800;
  int *range = hCom->data() + rangeStart;
  const int count = static_cast<int>(rangeEnd - rangeStart);
  if (ctx.rank == owner) {
    MPI_CHECK(MPI_Send(range,
                       count,
                       MPI_INT,
                       owner - 1,
                       kComRangeBackTagBase + owner,
                       ctx.comm));
  } else if (ctx.rank == owner - 1) {
    MPI_CHECK(MPI_Recv(range,
                       count,
                       MPI_INT,
                       owner,
                       kComRangeBackTagBase + owner,
                       ctx.comm,
                       MPI_STATUS_IGNORE));
  }
}

