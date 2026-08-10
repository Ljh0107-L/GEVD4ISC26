double localRelativeDeviceDiff(cublasHandle_t cublas,
                               const double *dLhs,
                               const double *dRhs,
                               size_t elems)
{
  if (elems == 0) {
    return 0.0;
  }
  if (elems > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const int n = static_cast<int>(elems);
  const double negOne = -1.0;
  double *dTmp = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dTmp, sizeof(double) * elems));
  CUBLAS_CHECK_LOCAL(cublasDcopy(cublas, n, dLhs, 1, dTmp, 1));
  CUBLAS_CHECK_LOCAL(cublasDaxpy(cublas, n, &negOne, dRhs, 1, dTmp, 1));
  double diffNorm = 0.0;
  double rhsNorm = 0.0;
  CUBLAS_CHECK_LOCAL(cublasDnrm2(cublas, n, dTmp, 1, &diffNorm));
  CUBLAS_CHECK_LOCAL(cublasDnrm2(cublas, n, dRhs, 1, &rhsNorm));
  EVD_CUDA_CHECK(evdFree(dTmp));
  return diffNorm / std::max(rhsNorm, 1.0);
}

void debugCompareDoubleSbrWithSingle(const DistContext &ctx,
                                     cusolverDnHandle_t cusolver,
                                     cublasHandle_t cublas,
                                     const char *inputFile,
                                     double *dA,
                                     double *dWLocal,
                                     double *dYLocal)
{
  const int debugCompare = envIntOrDefault("EVD_DEBUG_COMPARE_DOUBLE_SBR", 0);
  if (debugCompare == 0) {
    return;
  }
  const long maxN = envIntOrDefault("EVD_DEBUG_COMPARE_MAX_N", 4096);
  if (ctx.n > maxN) {
    if (ctx.rank == 0) {
      std::cout << "Skipping EVD_DEBUG_COMPARE_DOUBLE_SBR because n=" << ctx.n
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

  const bool suffixComm =
      envIntOrDefault("EVD_SBR_SUFFIX_COMM",
                      envIntOrDefault("EVD_STAGE_PIPELINE", 0) != 0 ? 1 : 0) != 0;
  const size_t elems = static_cast<size_t>(ctx.n) * ctx.localCols;
  const double localA = suffixComm ? 0.0 : localRelativeDeviceDiff(cublas, dA, dARef, elems);
  const double localW = localRelativeDeviceDiff(cublas, dWLocal, dWRef, elems);
  const double localY = localRelativeDeviceDiff(cublas, dYLocal, dYRef, elems);
  double globalA = 0.0;
  double globalW = 0.0;
  double globalY = 0.0;
  MPI_CHECK(MPI_Allreduce(&localA, &globalA, 1, MPI_DOUBLE, MPI_MAX, ctx.comm));
  MPI_CHECK(MPI_Allreduce(&localW, &globalW, 1, MPI_DOUBLE, MPI_MAX, ctx.comm));
  MPI_CHECK(MPI_Allreduce(&localY, &globalY, 1, MPI_DOUBLE, MPI_MAX, ctx.comm));
  if (ctx.rank == 0) {
    std::cout << "EVD_DEBUG_COMPARE_DOUBLE_SBR ";
    if (suffixComm) {
      std::cout << "full_A_skipped=1 ";
    } else {
      std::cout << "rel_diff_A=" << std::setprecision(17) << globalA << " ";
    }
    std::cout << std::setprecision(17)
              << "rel_diff_W=" << globalW
              << " rel_diff_Y=" << globalY << std::endl;
  }

  freeSbrWorkspace(&refWs);
  EVD_CUDA_CHECK(evdFree(dARef));
  EVD_CUDA_CHECK(evdFree(dWRef));
  EVD_CUDA_CHECK(evdFree(dYRef));
}

