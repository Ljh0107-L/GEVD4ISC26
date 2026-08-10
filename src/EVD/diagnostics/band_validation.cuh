void debugCompareDoubleBandWithSingle(const DistContext &ctx,
                                      cusolverDnHandle_t cusolver,
                                      cublasHandle_t cublas,
                                      const char *inputFile,
                                      double *dBand)
{
  const int debugCompare = envIntOrDefault("EVD_DEBUG_COMPARE_DOUBLE_BAND", 0);
  if (debugCompare == 0) {
    return;
  }
  const long maxN = envIntOrDefault("EVD_DEBUG_COMPARE_MAX_N", 4096);
  if (ctx.n > maxN) {
    if (ctx.rank == 0) {
      std::cout << "Skipping EVD_DEBUG_COMPARE_DOUBLE_BAND because n=" << ctx.n
                << " exceeds EVD_DEBUG_COMPARE_MAX_N=" << maxN << std::endl;
    }
    return;
  }

  double *dARef = nullptr;
  double *dWRef = nullptr;
  double *dYRef = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dARef, sizeof(double) * static_cast<size_t>(ctx.n) * ctx.localCols));
  EVD_CUDA_CHECK(evdMalloc(&dWRef, sizeof(double) * static_cast<size_t>(ctx.n) * ctx.localCols));
  EVD_CUDA_CHECK(evdMalloc(&dYRef, sizeof(double) * static_cast<size_t>(ctx.n) * ctx.localCols));
  EVD_CUDA_CHECK(cudaMemset(dWRef, 0, sizeof(double) * static_cast<size_t>(ctx.n) * ctx.localCols));
  EVD_CUDA_CHECK(cudaMemset(dYRef, 0, sizeof(double) * static_cast<size_t>(ctx.n) * ctx.localCols));
  loadLocalColumnsFromBinary(inputFile, dARef, ctx.n, ctx.colStart, ctx.localCols, ctx.rank);
  symmetrizeLowerToUpperDistributed(dARef,
                                    ctx.n,
                                    ctx.localCols,
                                    ctx.counts,
                                    ctx.displs,
                                    ctx.rank,
                                    ctx.comm);

  SbrWorkspace refWs;
  allocateSbrWorkspace(ctx, &refWs, false);
  MPI_CHECK(MPI_Barrier(ctx.comm));
  distributedSbr(ctx, cusolver, cublas, dARef, dWRef, dYRef, &refWs, false);
  EVD_CUDA_CHECK(cudaDeviceSynchronize());

  const long ldBand = 2 * ctx.b;
  double *dBandRef = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dBandRef, sizeof(double) * static_cast<size_t>(ldBand) * ctx.n));
  gatherBandToAll(ctx, dARef, dBandRef);

  const size_t bandElems = static_cast<size_t>(ldBand) * ctx.n;
  const double localBand = localRelativeDeviceDiff(cublas, dBand, dBandRef, bandElems);
  std::vector<double> band(static_cast<size_t>(ldBand) * ctx.n);
  std::vector<double> refBand(static_cast<size_t>(ldBand) * ctx.n);
  EVD_CUDA_CHECK(cudaMemcpy(band.data(), dBand, sizeof(double) * band.size(), cudaMemcpyDeviceToHost));
  EVD_CUDA_CHECK(cudaMemcpy(refBand.data(), dBandRef, sizeof(double) * refBand.size(), cudaMemcpyDeviceToHost));
  double localMaxAbs = 0.0;
  size_t localMaxIndex = 0;
  double ownedDiffSq = 0.0;
  double ownedRefSq = 0.0;
  double ownedMaxAbs = 0.0;
  size_t ownedMaxIndex = static_cast<size_t>(ctx.colStart) * static_cast<size_t>(ldBand);
  for (size_t i = 0; i < band.size(); ++i) {
    const double diff = std::abs(band[i] - refBand[i]);
    if (diff > localMaxAbs) {
      localMaxAbs = diff;
      localMaxIndex = i;
    }
  }
  const long ownedStart = std::max<long>(0, ctx.colStart);
  const long ownedEnd =
      std::max<long>(ownedStart, std::min<long>(ctx.n, ctx.colStart + ctx.localCols));
  for (long col = ownedStart; col < ownedEnd; ++col) {
    for (long row = 0; row < ldBand; ++row) {
      const size_t idx = static_cast<size_t>(col) * static_cast<size_t>(ldBand) +
                         static_cast<size_t>(row);
      const double diff = band[idx] - refBand[idx];
      ownedDiffSq += diff * diff;
      ownedRefSq += refBand[idx] * refBand[idx];
      const double absDiff = std::abs(diff);
      if (absDiff > ownedMaxAbs) {
        ownedMaxAbs = absDiff;
        ownedMaxIndex = idx;
      }
    }
  }
  const double localOwnedBand =
      (ownedRefSq > 0.0) ? std::sqrt(ownedDiffSq / ownedRefSq) : 0.0;
  for (int printRank = 0; printRank < ctx.size; ++printRank) {
    MPI_CHECK(MPI_Barrier(ctx.comm));
    if (ctx.rank == printRank) {
      const long localMaxRowOffset = static_cast<long>(localMaxIndex % static_cast<size_t>(ldBand));
      const long localMaxCol = static_cast<long>(localMaxIndex / static_cast<size_t>(ldBand));
      const long ownedMaxRowOffset = static_cast<long>(ownedMaxIndex % static_cast<size_t>(ldBand));
      const long ownedMaxCol = static_cast<long>(ownedMaxIndex / static_cast<size_t>(ldBand));
      std::cout << "EVD_DEBUG_COMPARE_DOUBLE_BAND rank=" << ctx.rank
                << " local_rel_diff_band=" << std::setprecision(17) << localBand
                << " local_max_abs=" << localMaxAbs
                << " local_max_row_offset=" << localMaxRowOffset
                << " local_max_col=" << localMaxCol
                << " local_value=" << band[localMaxIndex]
                << " local_ref_value=" << refBand[localMaxIndex]
                << " owned_rel_diff_band=" << localOwnedBand
                << " owned_max_abs=" << ownedMaxAbs
                << " owned_max_row_offset=" << ownedMaxRowOffset
                << " owned_max_col=" << ownedMaxCol
                << " owned_value=" << band[ownedMaxIndex]
                << " owned_ref_value=" << refBand[ownedMaxIndex]
                << std::endl;
    }
  }
  MPI_CHECK(MPI_Barrier(ctx.comm));
  double globalBand = 0.0;
  double globalOwnedBand = 0.0;
  double globalOwnedMaxAbs = 0.0;
  MPI_CHECK(MPI_Allreduce(&localBand, &globalBand, 1, MPI_DOUBLE, MPI_MAX, ctx.comm));
  MPI_CHECK(MPI_Allreduce(&localOwnedBand, &globalOwnedBand, 1, MPI_DOUBLE, MPI_MAX, ctx.comm));
  MPI_CHECK(MPI_Allreduce(&ownedMaxAbs, &globalOwnedMaxAbs, 1, MPI_DOUBLE, MPI_MAX, ctx.comm));
  if (ctx.rank == 0) {
    double maxAbs = 0.0;
    size_t maxIndex = 0;
    for (size_t i = 0; i < band.size(); ++i) {
      const double diff = std::abs(band[i] - refBand[i]);
      if (diff > maxAbs) {
        maxAbs = diff;
        maxIndex = i;
      }
    }
    const long maxRowOffset = static_cast<long>(maxIndex % static_cast<size_t>(ldBand));
    const long maxCol = static_cast<long>(maxIndex / static_cast<size_t>(ldBand));
    std::cout << "EVD_DEBUG_COMPARE_DOUBLE_BAND rel_diff_band=" << std::setprecision(17)
              << globalBand << " max_abs=" << maxAbs
              << " max_row_offset=" << maxRowOffset
              << " max_col=" << maxCol
              << " double_value=" << band[maxIndex]
              << " ref_value=" << refBand[maxIndex]
              << " owned_rel_diff_band=" << globalOwnedBand
              << " owned_max_abs=" << globalOwnedMaxAbs
              << std::endl;
  }

  EVD_CUDA_CHECK(evdFree(dBandRef));
  freeSbrWorkspace(&refWs);
  EVD_CUDA_CHECK(evdFree(dARef));
  EVD_CUDA_CHECK(evdFree(dWRef));
  EVD_CUDA_CHECK(evdFree(dYRef));
}
