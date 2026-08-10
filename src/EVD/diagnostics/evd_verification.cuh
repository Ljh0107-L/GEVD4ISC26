double verifyFinalOrthogonality(const DistContext &ctx,
                                cublasHandle_t cublas,
                                const double *dQFinal,
                                long ldQFinal)
{
  const double one = 1.0;
  const double zero = 0.0;
  double *dGram = nullptr;
  EVD_CUDA_CHECK(evdMalloc(&dGram, sizeof(double) * static_cast<size_t>(ctx.n) * ctx.n));
  CUBLAS_CHECK_LOCAL(cublasDgemm(cublas,
                                 CUBLAS_OP_T,
                                 CUBLAS_OP_N,
                                 static_cast<int>(ctx.n),
                                 static_cast<int>(ctx.n),
                                 static_cast<int>(ctx.qLocalCols),
                                 &one,
                                 dQFinal,
                                 static_cast<int>(ldQFinal),
                                 dQFinal,
                                 static_cast<int>(ldQFinal),
                                 &zero,
                                 dGram,
                                 static_cast<int>(ctx.n)));
  std::vector<double> localGram(static_cast<size_t>(ctx.n) * ctx.n);
  std::vector<double> globalGram(static_cast<size_t>(ctx.n) * ctx.n);
  EVD_CUDA_CHECK(cudaMemcpy(localGram.data(),
                   dGram,
                   sizeof(double) * localGram.size(),
                   cudaMemcpyDeviceToHost));
  EVD_CUDA_CHECK(evdFree(dGram));
  MPI_CHECK(MPI_Allreduce(localGram.data(),
                          globalGram.data(),
                          static_cast<int>(globalGram.size()),
                          MPI_DOUBLE,
                          MPI_SUM,
                          ctx.comm));
  long double ss = 0.0;
  for (long col = 0; col < ctx.n; ++col) {
    for (long row = 0; row < ctx.n; ++row) {
      long double value = globalGram[static_cast<size_t>(row) + static_cast<size_t>(col) * ctx.n];
      if (row == col) {
        value -= 1.0;
      }
      ss += value * value;
    }
  }
  return static_cast<double>(std::sqrt(ss) / static_cast<long double>(ctx.n));
}

std::vector<double> readFullMatrixForVerification(const char *fileName, long n)
{
  const size_t totalDoubles = static_cast<size_t>(n) * static_cast<size_t>(n);
  const size_t expectedBytes = totalDoubles * sizeof(double);
  std::ifstream file(fileName, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "failed to open verification input matrix: " << fileName << std::endl;
    ::gevd4isc26::evd::abortActiveCommunicator(1);
  }
  const std::streamoff fileBytes = file.tellg();
  if (fileBytes != static_cast<std::streamoff>(expectedBytes)) {
    std::cerr << "verification input matrix size mismatch: expected "
              << expectedBytes << " bytes, got " << fileBytes << std::endl;
    ::gevd4isc26::evd::abortActiveCommunicator(1);
  }
  file.seekg(0, std::ios::beg);
  std::vector<double> matrix(totalDoubles);
  file.read(reinterpret_cast<char *>(matrix.data()), static_cast<std::streamsize>(expectedBytes));
  if (file.gcount() != static_cast<std::streamsize>(expectedBytes)) {
    std::cerr << "short read from verification input matrix" << std::endl;
    ::gevd4isc26::evd::abortActiveCommunicator(1);
  }
  return matrix;
}

void symmetrizeStoredLowerTriangle(std::vector<double> *matrix, long n)
{
  for (long col = 0; col < n; ++col) {
    for (long row = col + 1; row < n; ++row) {
      (*matrix)[static_cast<size_t>(col) + static_cast<size_t>(row) * n] =
          (*matrix)[static_cast<size_t>(row) + static_cast<size_t>(col) * n];
    }
  }
}

std::pair<double, double> verifyBackwardResidual(const DistContext &ctx,
                                                 const char *inputFile,
                                                 const double *dQFinal,
                                                 long ldQFinal,
                                                 const std::vector<double> &lambda)
{
  const size_t localElems = static_cast<size_t>(ldQFinal) * static_cast<size_t>(ctx.n);
  std::vector<double> localQ(localElems);
  EVD_CUDA_CHECK(cudaMemcpy(localQ.data(), dQFinal, sizeof(double) * localElems, cudaMemcpyDeviceToHost));

  std::vector<int> recvCounts;
  std::vector<int> recvDispls;
  std::vector<double> gatheredQ;
  if (ctx.rank == 0) {
    recvCounts.resize(ctx.size);
    recvDispls.resize(ctx.size);
    int offset = 0;
    for (int r = 0; r < ctx.size; ++r) {
      recvCounts[r] = static_cast<int>(ctx.qCounts[r] * ctx.n);
      recvDispls[r] = offset;
      offset += recvCounts[r];
    }
    gatheredQ.resize(static_cast<size_t>(ctx.n) * static_cast<size_t>(ctx.n));
  }

  MPI_CHECK(MPI_Gatherv(localQ.data(),
                        static_cast<int>(localElems),
                        MPI_DOUBLE,
                        ctx.rank == 0 ? gatheredQ.data() : nullptr,
                        ctx.rank == 0 ? recvCounts.data() : nullptr,
                        ctx.rank == 0 ? recvDispls.data() : nullptr,
                        MPI_DOUBLE,
                        0,
                        ctx.comm));

  double residualQLQt = 0.0;
  double residualQtLQ = 0.0;
  if (ctx.rank == 0) {
    if (static_cast<long>(lambda.size()) != ctx.n) {
      std::cerr << "missing eigenvalues for backward residual verification" << std::endl;
      MPI_Abort(ctx.comm, 1);
    }

    std::vector<double> fullQ(static_cast<size_t>(ctx.n) * static_cast<size_t>(ctx.n));
    for (int r = 0; r < ctx.size; ++r) {
      const long rows = ctx.qCounts[r];
      const long rowStart = ctx.qDispls[r];
      const double *rankQ = gatheredQ.data() + recvDispls[r];
      for (long col = 0; col < ctx.n; ++col) {
        for (long row = 0; row < rows; ++row) {
          fullQ[static_cast<size_t>(rowStart + row) + static_cast<size_t>(col) * ctx.n] =
              rankQ[static_cast<size_t>(row) + static_cast<size_t>(col) * rows];
        }
      }
    }

    std::vector<double> originalA = readFullMatrixForVerification(inputFile, ctx.n);
    symmetrizeStoredLowerTriangle(&originalA, ctx.n);
    long double normASq = 0.0;
    for (double value : originalA) {
      const long double a = static_cast<long double>(value);
      normASq += a * a;
    }
    const double denom = static_cast<double>(ctx.n) * std::sqrt(static_cast<double>(normASq));

    auto residualFrom = [&](const std::vector<double> &reconstructed) {
      long double residualSq = 0.0;
      for (size_t i = 0; i < originalA.size(); ++i) {
        const long double diff = static_cast<long double>(reconstructed[i]) - static_cast<long double>(originalA[i]);
        residualSq += diff * diff;
      }
      return std::sqrt(static_cast<double>(residualSq)) / denom;
    };

    std::vector<double> weightedQ(fullQ.size());
    for (long col = 0; col < ctx.n; ++col) {
      const double eig = lambda[col];
      for (long row = 0; row < ctx.n; ++row) {
        weightedQ[static_cast<size_t>(row) + static_cast<size_t>(col) * ctx.n] =
            fullQ[static_cast<size_t>(row) + static_cast<size_t>(col) * ctx.n] * eig;
      }
    }

    std::vector<double> reconstructedQLQt(fullQ.size());
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasTrans,
                static_cast<int>(ctx.n),
                static_cast<int>(ctx.n),
                static_cast<int>(ctx.n),
                1.0,
                weightedQ.data(),
                static_cast<int>(ctx.n),
                fullQ.data(),
                static_cast<int>(ctx.n),
                0.0,
                reconstructedQLQt.data(),
                static_cast<int>(ctx.n));
    residualQLQt = residualFrom(reconstructedQLQt);

    std::vector<double> rowWeightedQ(fullQ.size());
    for (long col = 0; col < ctx.n; ++col) {
      for (long row = 0; row < ctx.n; ++row) {
        rowWeightedQ[static_cast<size_t>(row) + static_cast<size_t>(col) * ctx.n] =
            fullQ[static_cast<size_t>(row) + static_cast<size_t>(col) * ctx.n] * lambda[row];
      }
    }

    std::vector<double> reconstructedQtLQ(fullQ.size());
    cblas_dgemm(CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                static_cast<int>(ctx.n),
                static_cast<int>(ctx.n),
                static_cast<int>(ctx.n),
                1.0,
                fullQ.data(),
                static_cast<int>(ctx.n),
                rowWeightedQ.data(),
                static_cast<int>(ctx.n),
                0.0,
                reconstructedQtLQ.data(),
                static_cast<int>(ctx.n));
    residualQtLQ = residualFrom(reconstructedQtLQ);
  }

  double residuals[2] = {residualQLQt, residualQtLQ};
  MPI_CHECK(MPI_Bcast(residuals, 2, MPI_DOUBLE, 0, ctx.comm));
  return {residuals[0], residuals[1]};
}

void compareEigenvaluesWithInput(long n,
                                 const char *inputFile,
                                 const std::vector<double> &lambda)
{
  if (static_cast<long>(lambda.size()) != n) {
    std::cerr << "missing eigenvalues for input eigenvalue comparison" << std::endl;
    ::gevd4isc26::evd::abortActiveCommunicator(1);
  }

  std::vector<double> dense = readFullMatrixForVerification(inputFile, n);
  symmetrizeStoredLowerTriangle(&dense, n);
  std::vector<double> reference(static_cast<size_t>(n));
  const lapack_int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR,
                                         'N',
                                         'L',
                                         static_cast<lapack_int>(n),
                                         dense.data(),
                                         static_cast<lapack_int>(n),
                                         reference.data());
  if (info != 0) {
    std::cerr << "LAPACKE_dsyevd input eigenvalue comparison failed with info="
              << info << std::endl;
    ::gevd4isc26::evd::abortActiveCommunicator(1);
  }

  auto computeDiff = [n](const std::vector<double> &lhs,
                         const std::vector<double> &rhs,
                         double *maxAbs,
                         double *rms) {
    long double sq = 0.0;
    *maxAbs = 0.0;
    for (long i = 0; i < n; ++i) {
      const double diff = std::abs(lhs[static_cast<size_t>(i)] - rhs[static_cast<size_t>(i)]);
      *maxAbs = std::max(*maxAbs, diff);
      sq += static_cast<long double>(diff) * diff;
    }
    *rms = std::sqrt(static_cast<double>(sq / static_cast<long double>(n)));
  };

  double maxAbs = 0.0;
  double rms = 0.0;
  computeDiff(reference, lambda, &maxAbs, &rms);

  std::vector<double> sortedLambda = lambda;
  std::sort(sortedLambda.begin(), sortedLambda.end());
  double sortedMaxAbs = 0.0;
  double sortedRms = 0.0;
  computeDiff(reference, sortedLambda, &sortedMaxAbs, &sortedRms);

  double scale = 1.0;
  for (long i = 0; i < n; ++i) {
    scale = std::max(scale, std::abs(reference[static_cast<size_t>(i)]));
  }
  std::cout << std::setprecision(17)
            << "Input DSYEVD eig first=" << reference.front()
            << " last=" << reference.back()
            << " max_abs_diff=" << maxAbs
            << " rms_diff=" << rms
            << " rel_max_diff=" << (maxAbs / scale)
            << " sorted_max_abs_diff=" << sortedMaxAbs
            << " sorted_rms_diff=" << sortedRms
            << " sorted_rel_max_diff=" << (sortedMaxAbs / scale)
            << std::endl;
}
