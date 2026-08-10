std::vector<long> buildPackedUOffsetsForBack(long n)
{
  const int sweepCount = static_cast<int>((n - 2 + (kBcBackSweepRows - 1)) / kBcBackSweepRows);
  const long lastSweepUCount = n - (static_cast<long>(sweepCount - 1) * kBcBackSweepRows + 1) - 1;
  std::vector<long> offsets(sweepCount);
  long offset = 0;
  for (int i = 0; i < sweepCount; ++i) {
    const long totalU = lastSweepUCount + static_cast<long>(i) * kBcBackSweepRows;
    offsets[i] = offset;
    const long paddedU = ((totalU + kBcBackPackedUColumnTile - 1) / kBcBackPackedUColumnTile) * kBcBackPackedUColumnTile;
    offset += paddedU * kBcBackSweepRows;
  }
  return offsets;
}

struct PackedUOffsetTable {
  std::vector<long> host;
  long *device = nullptr;
};

PackedUOffsetTable createPackedUOffsetTable(long n)
{
  PackedUOffsetTable table;
  table.host = buildPackedUOffsetsForBack(n);
  EVD_CUDA_CHECK(evdMalloc(&table.device, sizeof(long) * table.host.size()));
  EVD_CUDA_CHECK(cudaMemcpy(table.device,
                   table.host.data(),
                   sizeof(long) * table.host.size(),
                   cudaMemcpyHostToDevice));
  return table;
}

void destroyPackedUOffsetTable(PackedUOffsetTable *table)
{
  if (table->device != nullptr) {
    EVD_CUDA_CHECK(evdFree(table->device));
    table->device = nullptr;
  }
  table->host.clear();
}

void runBcOnReplicatedBandPackedUFull(const DistContext &ctx,
                                      double *dBand,
                                      double *dUPacked,
                                      long packedUElems,
                                      const PackedUOffsetTable &packedUOffsets)
{
  if (ctx.b != kBcBandwidth) {
    std::cerr << "EVD currently instantiates the existing BC kernel for b=32; got b="
              << ctx.b << std::endl;
    MPI_Abort(ctx.comm, 1);
  }

  int zero = 0;
  EVD_CUDA_CHECK(cudaMemcpyToSymbol(bcStopFlag, &zero, sizeof(int)));
  int *com = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&com, sizeof(int) * ctx.n));
  EVD_CUDA_CHECK(cudaMemset(com, 0, sizeof(int) * ctx.n));
  EVD_CUDA_CHECK(cudaMemset(dUPacked, 0, sizeof(double) * static_cast<size_t>(packedUElems)));

  long *dPackedUOffsets = packedUOffsets.device;
  int dev = 0;
  EVD_CUDA_CHECK(cudaGetDevice(&dev));
  int numBlocksPerSm = 0;
  int numThreads = 32 * 32;
  cudaDeviceProp deviceProp;
  EVD_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  EVD_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                      chaseBulgesPackedKernel<kBcBandwidth>,
                                                      numThreads,
                                                      0));
  int blockNum = numBlocksPerSm * deviceProp.multiProcessorCount;
  int nInt = static_cast<int>(ctx.n);
  int bInt = static_cast<int>(ctx.b);
  int ldBandInt = static_cast<int>(2 * ctx.b);
  int packedUSweepCount = static_cast<int>(packedUOffsets.host.size());
  void *kernelArgs[] = {
      (void *)&nInt,
      (void *)&bInt,
      (void *)&dBand,
      (void *)&ldBandInt,
      (void *)&dUPacked,
      (void *)&dPackedUOffsets,
      (void *)&packedUSweepCount,
      (void *)&blockNum,
      (void *)&com,
  };
  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(blockNum, 1, 1);
  EVD_CUDA_CHECK(cudaLaunchCooperativeKernel((void *)chaseBulgesPackedKernel<kBcBandwidth>,
                                    dimGrid,
                                    dimBlock,
                                    kernelArgs));
  EVD_CUDA_CHECK(cudaGetLastError());
  EVD_CUDA_CHECK(cudaDeviceSynchronize());
  EVD_CUDA_CHECK(evdFree(com));
}

void broadcastPackedUOwnerSlices(const DistContext &ctx,
                                 double *dUPacked,
                                 const std::vector<long> &packedUOffsets,
                                 long ownerSweepStart,
                                 long ownerSweepEnd,
                                 int owner)
{
  const int sweepCount = static_cast<int>(packedUOffsets.size());
  const long lastSweepUCount = ctx.n - (static_cast<long>(sweepCount - 1) * kBcBackSweepRows + 1) - 1;
  bool hasSlice = false;
  for (int sweepIndex = 0; sweepIndex < sweepCount; ++sweepIndex) {
    const long totalU = lastSweepUCount + static_cast<long>(sweepIndex) * kBcBackSweepRows;
    const long sliceStart = std::max<long>(0, ownerSweepStart);
    const long sliceEnd = std::min<long>(ownerSweepEnd, totalU);
    if (sliceEnd > sliceStart) {
      hasSlice = true;
      break;
    }
  }
  if (!hasSlice) {
    return;
  }

  NCCL_CHECK_LOCAL(ncclGroupStart());
  for (int sweepIndex = 0; sweepIndex < sweepCount; ++sweepIndex) {
    const long totalU = lastSweepUCount + static_cast<long>(sweepIndex) * kBcBackSweepRows;
    const long sliceStart = std::max<long>(0, ownerSweepStart);
    const long sliceEnd = std::min<long>(ownerSweepEnd, totalU);
    if (sliceEnd <= sliceStart) {
      continue;
    }
    double *slice = dUPacked + packedUOffsets[sweepIndex] + sliceStart * kBcBackSweepRows;
    NCCL_CHECK_LOCAL(ncclBroadcast(slice,
                                   slice,
                                   static_cast<size_t>(sliceEnd - sliceStart) * kBcBackSweepRows,
                                   ncclDouble,
                                   owner,
                                   ctx.nccl,
                                   ctx.commStream));
  }
  NCCL_CHECK_LOCAL(ncclGroupEnd());
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
}

struct PackedUSlicePlan {
  std::vector<long> srcOffsets;
  std::vector<long> dstOffsets;
  std::vector<long> lengths;
  long totalElems = 0;
};

PackedUSlicePlan buildPackedUOwnerSlicePlan(const DistContext &ctx,
                                            const std::vector<long> &packedUOffsets,
                                            long ownerSweepStart,
                                            long ownerSweepEnd)
{
  PackedUSlicePlan plan;
  const int sweepCount = static_cast<int>(packedUOffsets.size());
  const long lastSweepUCount = ctx.n - (static_cast<long>(sweepCount - 1) * kBcBackSweepRows + 1) - 1;
  for (int sweepIndex = 0; sweepIndex < sweepCount; ++sweepIndex) {
    const long totalU = lastSweepUCount + static_cast<long>(sweepIndex) * kBcBackSweepRows;
    const long sliceStart = std::max<long>(0, ownerSweepStart);
    const long sliceEnd = std::min<long>(ownerSweepEnd, totalU);
    if (sliceEnd <= sliceStart) {
      continue;
    }
    const long length = (sliceEnd - sliceStart) * kBcBackSweepRows;
    plan.srcOffsets.push_back(packedUOffsets[sweepIndex] + sliceStart * kBcBackSweepRows);
    plan.dstOffsets.push_back(plan.totalElems);
    plan.lengths.push_back(length);
    plan.totalElems += length;
  }
  return plan;
}

void copyPackedUSlicesAsync(double *dst,
                            const double *src,
                            const PackedUSlicePlan &plan,
                            bool pack,
                            cudaStream_t stream)
{
  for (size_t i = 0; i < plan.lengths.size(); ++i) {
    double *dstPtr = pack ? (dst + plan.dstOffsets[i]) : (dst + plan.srcOffsets[i]);
    const double *srcPtr = pack ? (src + plan.srcOffsets[i]) : (src + plan.dstOffsets[i]);
    EVD_CUDA_CHECK(cudaMemcpyAsync(dstPtr,
                          srcPtr,
                          sizeof(double) * static_cast<size_t>(plan.lengths[i]),
                          cudaMemcpyDeviceToDevice,
                          stream));
  }
}

void broadcastPackedUOwnerSlicesCompact(const DistContext &ctx,
                                        double *dUPacked,
                                        double *dCompact,
                                        const PackedUSlicePlan &plan,
                                        int owner)
{
  if (plan.totalElems <= 0) {
    return;
  }
  if (ctx.rank == owner) {
    copyPackedUSlicesAsync(dCompact, dUPacked, plan, true, ctx.commStream);
    EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
  }
  NCCL_CHECK_LOCAL(ncclBroadcast(dCompact,
                                 dCompact,
                                 static_cast<size_t>(plan.totalElems),
                                 ncclDouble,
                                 owner,
                                 ctx.nccl,
                                 ctx.commStream));
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
  copyPackedUSlicesAsync(dUPacked, dCompact, plan, false, ctx.commStream);
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
}

void handoffPackedUSlicesToNextOwnerCompact(const DistContext &ctx,
                                            double *dUPacked,
                                            double *dCompact,
                                            const PackedUSlicePlan &plan,
                                            int owner)
{
  if (owner + 1 >= ctx.size || plan.totalElems <= 0) {
    return;
  }
  if (debugStageProgressEnabled() && (ctx.rank == owner || ctx.rank == owner + 1)) {
    std::cout << "StageProgress rank=" << ctx.rank
              << " phase=packedu_handoff_compact owner=" << owner
              << " dst=" << (owner + 1)
              << " elems=" << plan.totalElems << std::endl;
  }
  if (ctx.rank == owner) {
    copyPackedUSlicesAsync(dCompact, dUPacked, plan, true, ctx.commStream);
    EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
  }
  if (ctx.rank == owner || ctx.rank == owner + 1) {
    NCCL_CHECK_LOCAL(ncclGroupStart());
    if (ctx.rank == owner) {
      NCCL_CHECK_LOCAL(ncclSend(dCompact,
                                static_cast<size_t>(plan.totalElems),
                                ncclDouble,
                                owner + 1,
                                ctx.nccl,
                                ctx.commStream));
    } else {
      NCCL_CHECK_LOCAL(ncclRecv(dCompact,
                                static_cast<size_t>(plan.totalElems),
                                ncclDouble,
                                owner,
                                ctx.nccl,
                                ctx.commStream));
    }
    NCCL_CHECK_LOCAL(ncclGroupEnd());
    EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
  }
  if (ctx.rank == owner + 1) {
    copyPackedUSlicesAsync(dUPacked, dCompact, plan, false, ctx.commStream);
    EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));
  }
}

struct PackedUProgressPlan {
  std::vector<int> sweeps;
  std::vector<int> rowStarts;
  std::vector<int> lengths;
  std::vector<long> dstOffsets;
  long totalElems = 0;
};

struct PackedUProgressDeviceWorkspace {
  int *dSweeps = nullptr;
  int *dRowStarts = nullptr;
  int *dLengths = nullptr;
  long *dDstOffsets = nullptr;
  size_t capacity = 0;
};

void releasePackedUProgressDeviceWorkspace(PackedUProgressDeviceWorkspace *ws)
{
  if (ws == nullptr) {
    return;
  }
  if (ws->dSweeps != nullptr) {
    EVD_CUDA_CHECK(evdFree(ws->dSweeps));
  }
  if (ws->dRowStarts != nullptr) {
    EVD_CUDA_CHECK(evdFree(ws->dRowStarts));
  }
  if (ws->dLengths != nullptr) {
    EVD_CUDA_CHECK(evdFree(ws->dLengths));
  }
  if (ws->dDstOffsets != nullptr) {
    EVD_CUDA_CHECK(evdFree(ws->dDstOffsets));
  }
  ws->dSweeps = nullptr;
  ws->dRowStarts = nullptr;
  ws->dLengths = nullptr;
  ws->dDstOffsets = nullptr;
  ws->capacity = 0;
}

void ensurePackedUProgressDeviceWorkspace(PackedUProgressDeviceWorkspace *ws,
                                          size_t segmentCount)
{
  if (ws == nullptr || ws->capacity >= segmentCount) {
    return;
  }
  releasePackedUProgressDeviceWorkspace(ws);
  EVD_CUDA_CHECK(evdMalloc(&ws->dSweeps, sizeof(int) * segmentCount));
  EVD_CUDA_CHECK(evdMalloc(&ws->dRowStarts, sizeof(int) * segmentCount));
  EVD_CUDA_CHECK(evdMalloc(&ws->dLengths, sizeof(int) * segmentCount));
  EVD_CUDA_CHECK(evdMalloc(&ws->dDstOffsets, sizeof(long) * segmentCount));
  ws->capacity = segmentCount;
}

long clampPackedUProgressRow(long progress, long sweep, long n, long b)
{
  if (progress >= n + 3 * b) {
    return n;
  }
  return std::max<long>(sweep + 1, std::min<long>(n, progress));
}

PackedUProgressPlan buildPackedUProgressPlan(const DistContext &ctx,
                                             const std::vector<int> &before,
                                             const std::vector<int> &after)
{
  PackedUProgressPlan plan;
  const long totalSweeps = std::max<long>(0, ctx.n - 2);
  plan.sweeps.reserve(static_cast<size_t>(totalSweeps));
  plan.rowStarts.reserve(static_cast<size_t>(totalSweeps));
  plan.lengths.reserve(static_cast<size_t>(totalSweeps));
  plan.dstOffsets.reserve(static_cast<size_t>(totalSweeps));
  for (long sweep = 0; sweep < totalSweeps; ++sweep) {
    const long rowStart =
        clampPackedUProgressRow(before[static_cast<size_t>(sweep)], sweep, ctx.n, ctx.b);
    const long rowEnd =
        clampPackedUProgressRow(after[static_cast<size_t>(sweep)], sweep, ctx.n, ctx.b);
    if (rowEnd <= rowStart) {
      continue;
    }
    plan.sweeps.push_back(static_cast<int>(sweep));
    plan.rowStarts.push_back(static_cast<int>(rowStart));
    plan.lengths.push_back(static_cast<int>(rowEnd - rowStart));
    plan.dstOffsets.push_back(plan.totalElems);
    plan.totalElems += rowEnd - rowStart;
  }
  return plan;
}

PackedUProgressPlan buildPackedURowOwnerPlan(const DistContext &ctx,
                                             long rowStart,
                                             long rowEnd)
{
  PackedUProgressPlan plan;
  const long totalSweeps = std::max<long>(0, ctx.n - 2);
  const long boundedRowStart = std::max<long>(0, std::min<long>(ctx.n, rowStart));
  const long boundedRowEnd = std::max<long>(boundedRowStart, std::min<long>(ctx.n, rowEnd));
  plan.sweeps.reserve(static_cast<size_t>(totalSweeps));
  plan.rowStarts.reserve(static_cast<size_t>(totalSweeps));
  plan.lengths.reserve(static_cast<size_t>(totalSweeps));
  plan.dstOffsets.reserve(static_cast<size_t>(totalSweeps));
  for (long sweep = 0; sweep < totalSweeps; ++sweep) {
    const long uStart = std::max<long>(sweep + 1, boundedRowStart);
    const long uEnd = boundedRowEnd;
    if (uEnd <= uStart) {
      continue;
    }
    plan.sweeps.push_back(static_cast<int>(sweep));
    plan.rowStarts.push_back(static_cast<int>(uStart));
    plan.lengths.push_back(static_cast<int>(uEnd - uStart));
    plan.dstOffsets.push_back(plan.totalElems);
    plan.totalElems += uEnd - uStart;
  }
  return plan;
}

__global__ void packPackedUProgressKernel(const double *packedU,
                                          double *compact,
                                          const int *sweeps,
                                          const int *rowStarts,
                                          const int *lengths,
                                          const long *dstOffsets,
                                          const long *packedUOffsets,
                                          int segmentCount,
                                          int packedUSweepCount,
                                          int n)
{
  const int segment = blockIdx.x;
  if (segment >= segmentCount) {
    return;
  }
  const long sweep = sweeps[segment];
  const long rowStart = rowStarts[segment];
  const long length = lengths[segment];
  const long dstBase = dstOffsets[segment];
  for (long i = threadIdx.x; i < length; i += blockDim.x) {
    const long row = rowStart + i;
    const long delta = row - sweep - 1;
    if (delta < 0 || row >= n) {
      continue;
    }
    const long sweepBaseRow =
        (delta / kBcBackSweepRows) * kBcBackSweepRows;
    const int sweepIndex =
        packedUSweepCount - 1 - static_cast<int>(sweepBaseRow / kBcBackSweepRows);
    const long lane = delta - sweepBaseRow;
    const long totalU = static_cast<long>(n) - sweepBaseRow - 2;
    if (sweepIndex >= 0 && sweepIndex < packedUSweepCount &&
        lane < kBcBackSweepRows && sweep < totalU) {
      const long src = packedUOffsets[sweepIndex] +
                       sweep * kBcBackSweepRows + lane;
      compact[dstBase + i] = packedU[src];
    }
  }
}

__global__ void unpackPackedUProgressKernel(double *packedU,
                                            const double *compact,
                                            const int *sweeps,
                                            const int *rowStarts,
                                            const int *lengths,
                                            const long *dstOffsets,
                                            const long *packedUOffsets,
                                            int segmentCount,
                                            int packedUSweepCount,
                                            int n)
{
  const int segment = blockIdx.x;
  if (segment >= segmentCount) {
    return;
  }
  const long sweep = sweeps[segment];
  const long rowStart = rowStarts[segment];
  const long length = lengths[segment];
  const long srcBase = dstOffsets[segment];
  for (long i = threadIdx.x; i < length; i += blockDim.x) {
    const long row = rowStart + i;
    const long delta = row - sweep - 1;
    if (delta < 0 || row >= n) {
      continue;
    }
    const long sweepBaseRow =
        (delta / kBcBackSweepRows) * kBcBackSweepRows;
    const int sweepIndex =
        packedUSweepCount - 1 - static_cast<int>(sweepBaseRow / kBcBackSweepRows);
    const long lane = delta - sweepBaseRow;
    const long totalU = static_cast<long>(n) - sweepBaseRow - 2;
    if (sweepIndex >= 0 && sweepIndex < packedUSweepCount &&
        lane < kBcBackSweepRows && sweep < totalU) {
      const long dst = packedUOffsets[sweepIndex] +
                       sweep * kBcBackSweepRows + lane;
      packedU[dst] = compact[srcBase + i];
    }
  }
}

void broadcastPackedUProgressCompact(const DistContext &ctx,
                                     double *dUPacked,
                                     double *dCompact,
                                     const PackedUProgressPlan &plan,
                                     const long *dPackedUOffsets,
                                     int packedUSweepCount,
                                     int owner,
                                     PackedUProgressDeviceWorkspace *workspace = nullptr)
{
  const int segmentCount = static_cast<int>(plan.sweeps.size());
  if (segmentCount <= 0 || plan.totalElems <= 0) {
    return;
  }
  PackedUProgressDeviceWorkspace localWorkspace;
  PackedUProgressDeviceWorkspace *ws =
      (workspace != nullptr) ? workspace : &localWorkspace;
  ensurePackedUProgressDeviceWorkspace(ws, static_cast<size_t>(segmentCount));
  EVD_CUDA_CHECK(cudaMemcpyAsync(ws->dSweeps,
                        plan.sweeps.data(),
                        sizeof(int) * static_cast<size_t>(segmentCount),
                        cudaMemcpyHostToDevice,
                        ctx.commStream));
  EVD_CUDA_CHECK(cudaMemcpyAsync(ws->dRowStarts,
                        plan.rowStarts.data(),
                        sizeof(int) * static_cast<size_t>(segmentCount),
                        cudaMemcpyHostToDevice,
                        ctx.commStream));
  EVD_CUDA_CHECK(cudaMemcpyAsync(ws->dLengths,
                        plan.lengths.data(),
                        sizeof(int) * static_cast<size_t>(segmentCount),
                        cudaMemcpyHostToDevice,
                        ctx.commStream));
  EVD_CUDA_CHECK(cudaMemcpyAsync(ws->dDstOffsets,
                        plan.dstOffsets.data(),
                        sizeof(long) * static_cast<size_t>(segmentCount),
                        cudaMemcpyHostToDevice,
                        ctx.commStream));

  const int threads = 256;
  if (ctx.rank == owner) {
    packPackedUProgressKernel<<<segmentCount, threads, 0, ctx.commStream>>>(
        dUPacked,
        dCompact,
        ws->dSweeps,
        ws->dRowStarts,
        ws->dLengths,
        ws->dDstOffsets,
        dPackedUOffsets,
        segmentCount,
	        packedUSweepCount,
	        static_cast<int>(ctx.n));
    EVD_CUDA_CHECK(cudaGetLastError());
  }

  NCCL_CHECK_LOCAL(ncclBroadcast(dCompact,
                                 dCompact,
                                 static_cast<size_t>(plan.totalElems),
                                 ncclDouble,
                                 owner,
                                 ctx.nccl,
                                 ctx.commStream));

  unpackPackedUProgressKernel<<<segmentCount, threads, 0, ctx.commStream>>>(
      dUPacked,
      dCompact,
      ws->dSweeps,
      ws->dRowStarts,
      ws->dLengths,
      ws->dDstOffsets,
      dPackedUOffsets,
      segmentCount,
      packedUSweepCount,
      static_cast<int>(ctx.n));
  EVD_CUDA_CHECK(cudaGetLastError());
  EVD_CUDA_CHECK(cudaStreamSynchronize(ctx.commStream));

  if (workspace == nullptr) {
    releasePackedUProgressDeviceWorkspace(&localWorkspace);
  }
}

void broadcastPackedUProgressPlanMetadata(const DistContext &ctx,
                                          PackedUProgressPlan *plan,
                                          int owner)
{
  int segmentCount =
      (ctx.rank == owner) ? static_cast<int>(plan->sweeps.size()) : 0;
  long totalElems = (ctx.rank == owner) ? plan->totalElems : 0;
  MPI_CHECK(MPI_Bcast(&segmentCount, 1, MPI_INT, owner, ctx.comm));
  MPI_CHECK(MPI_Bcast(&totalElems, 1, MPI_LONG, owner, ctx.comm));
  if (ctx.rank != owner) {
    plan->sweeps.resize(static_cast<size_t>(segmentCount));
    plan->rowStarts.resize(static_cast<size_t>(segmentCount));
    plan->lengths.resize(static_cast<size_t>(segmentCount));
    plan->dstOffsets.resize(static_cast<size_t>(segmentCount));
  }
  plan->totalElems = totalElems;
  if (segmentCount <= 0) {
    return;
  }
  MPI_CHECK(MPI_Bcast(plan->sweeps.data(), segmentCount, MPI_INT, owner, ctx.comm));
  MPI_CHECK(MPI_Bcast(plan->rowStarts.data(), segmentCount, MPI_INT, owner, ctx.comm));
  MPI_CHECK(MPI_Bcast(plan->lengths.data(), segmentCount, MPI_INT, owner, ctx.comm));
  MPI_CHECK(MPI_Bcast(plan->dstOffsets.data(), segmentCount, MPI_LONG, owner, ctx.comm));
}

void appendPackedUProgressPlan(PackedUProgressPlan *dst,
                               const PackedUProgressPlan &src)
{
  if (src.totalElems <= 0 || src.sweeps.empty()) {
    return;
  }
  const long baseOffset = dst->totalElems;
  dst->sweeps.insert(dst->sweeps.end(), src.sweeps.begin(), src.sweeps.end());
  dst->rowStarts.insert(dst->rowStarts.end(), src.rowStarts.begin(), src.rowStarts.end());
  dst->lengths.insert(dst->lengths.end(), src.lengths.begin(), src.lengths.end());
  dst->dstOffsets.reserve(dst->dstOffsets.size() + src.dstOffsets.size());
  for (long offset : src.dstOffsets) {
    dst->dstOffsets.push_back(baseOffset + offset);
  }
  dst->totalElems += src.totalElems;
}

void coalescePackedUProgressPlan(PackedUProgressPlan *plan)
{
  if (plan == nullptr || plan->sweeps.size() <= 1) {
    return;
  }
  struct Segment {
    int sweep;
    int rowStart;
    int rowEnd;
  };
  std::vector<Segment> segments;
  segments.reserve(plan->sweeps.size());
  for (size_t i = 0; i < plan->sweeps.size(); ++i) {
    const int rowStart = plan->rowStarts[i];
    const int rowEnd = rowStart + plan->lengths[i];
    if (rowEnd > rowStart) {
      segments.push_back({plan->sweeps[i], rowStart, rowEnd});
    }
  }
  if (segments.empty()) {
    plan->sweeps.clear();
    plan->rowStarts.clear();
    plan->lengths.clear();
    plan->dstOffsets.clear();
    plan->totalElems = 0;
    return;
  }
  std::sort(segments.begin(),
            segments.end(),
            [](const Segment &a, const Segment &b) {
              if (a.sweep != b.sweep) {
                return a.sweep < b.sweep;
              }
              return a.rowStart < b.rowStart;
            });

  plan->sweeps.clear();
  plan->rowStarts.clear();
  plan->lengths.clear();
  plan->dstOffsets.clear();
  plan->totalElems = 0;

  Segment current = segments.front();
  auto flushCurrent = [&]() {
    plan->sweeps.push_back(current.sweep);
    plan->rowStarts.push_back(current.rowStart);
    plan->lengths.push_back(current.rowEnd - current.rowStart);
    plan->dstOffsets.push_back(plan->totalElems);
    plan->totalElems += current.rowEnd - current.rowStart;
  };
  for (size_t i = 1; i < segments.size(); ++i) {
    const Segment &next = segments[i];
    if (next.sweep == current.sweep && next.rowStart <= current.rowEnd) {
      current.rowEnd = std::max(current.rowEnd, next.rowEnd);
      continue;
    }
    flushCurrent();
    current = next;
  }
  flushCurrent();
}

