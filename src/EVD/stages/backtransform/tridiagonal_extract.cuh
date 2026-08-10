void extractTridiagonalFromBand(long n, long ldBand, double *dBand, std::vector<double> *diag, std::vector<double> *offdiag)
{
  std::vector<double> hBand(static_cast<size_t>(ldBand) * n);
  EVD_CUDA_CHECK(cudaMemcpy(hBand.data(), dBand, sizeof(double) * hBand.size(), cudaMemcpyDeviceToHost));
  diag->resize(n);
  offdiag->resize(n - 1);
  for (long j = 0; j < n; ++j) {
    (*diag)[j] = hBand[j * ldBand];
    if (j < n - 1) {
      (*offdiag)[j] = hBand[1 + j * ldBand];
    }
  }
}

void extractTridiagonalOwnedColumnsToRoot(const DistContext &ctx,
                                          long ldBand,
                                          double *dBand,
                                          std::vector<double> *diag,
                                          std::vector<double> *offdiag)
{
  std::vector<double> hLocalBand(static_cast<size_t>(ldBand) * static_cast<size_t>(ctx.localCols));
  EVD_CUDA_CHECK(cudaMemcpy(hLocalBand.data(),
                   dBand + ctx.colStart * ldBand,
                   sizeof(double) * hLocalBand.size(),
                   cudaMemcpyDeviceToHost));

  std::vector<double> localDiag(ctx.localCols);
  std::vector<double> localOff(ctx.localCols);
  for (long localCol = 0; localCol < ctx.localCols; ++localCol) {
    const long globalCol = ctx.colStart + localCol;
    localDiag[localCol] = hLocalBand[static_cast<size_t>(localCol) * ldBand];
    localOff[localCol] =
        (globalCol < ctx.n - 1) ? hLocalBand[static_cast<size_t>(localCol) * ldBand + 1] : 0.0;
  }

  std::vector<int> counts(ctx.size);
  std::vector<int> displs(ctx.size);
  for (int r = 0; r < ctx.size; ++r) {
    counts[r] = static_cast<int>(ctx.counts[r]);
    displs[r] = static_cast<int>(ctx.displs[r]);
  }

  std::vector<double> gatheredDiag;
  std::vector<double> gatheredOff;
  if (ctx.rank == 0) {
    gatheredDiag.resize(ctx.n);
    gatheredOff.resize(ctx.n);
  }

  MPI_CHECK(MPI_Gatherv(localDiag.data(),
                        static_cast<int>(localDiag.size()),
                        MPI_DOUBLE,
                        ctx.rank == 0 ? gatheredDiag.data() : nullptr,
                        counts.data(),
                        displs.data(),
                        MPI_DOUBLE,
                        0,
                        ctx.comm));
  MPI_CHECK(MPI_Gatherv(localOff.data(),
                        static_cast<int>(localOff.size()),
                        MPI_DOUBLE,
                        ctx.rank == 0 ? gatheredOff.data() : nullptr,
                        counts.data(),
                        displs.data(),
                        MPI_DOUBLE,
                        0,
                        ctx.comm));

  if (ctx.rank == 0) {
    *diag = std::move(gatheredDiag);
    offdiag->assign(gatheredOff.begin(), gatheredOff.begin() + (ctx.n - 1));
  }
}

void extractTridiagonalFromBandOwnerToRoot(const DistContext &ctx,
                                           long ldBand,
                                           double *dBand,
                                           int owner,
                                           std::vector<double> *diag,
                                           std::vector<double> *offdiag)
{
  constexpr int kDiagTag = 7311;
  constexpr int kOffdiagTag = 7312;

  if (ctx.rank == owner) {
    std::vector<double> hBand(static_cast<size_t>(ldBand) * static_cast<size_t>(ctx.n));
    EVD_CUDA_CHECK(cudaMemcpy(hBand.data(), dBand, sizeof(double) * hBand.size(), cudaMemcpyDeviceToHost));

    std::vector<double> ownerDiag(ctx.n);
    std::vector<double> ownerOff(std::max<long>(0, ctx.n - 1));
    for (long col = 0; col < ctx.n; ++col) {
      ownerDiag[static_cast<size_t>(col)] = hBand[static_cast<size_t>(col) * ldBand];
      if (col < ctx.n - 1) {
        ownerOff[static_cast<size_t>(col)] = hBand[static_cast<size_t>(col) * ldBand + 1];
      }
    }

    if (owner == 0) {
      *diag = std::move(ownerDiag);
      *offdiag = std::move(ownerOff);
    } else {
      MPI_CHECK(MPI_Send(ownerDiag.data(), static_cast<int>(ownerDiag.size()), MPI_DOUBLE, 0, kDiagTag, ctx.comm));
      MPI_CHECK(MPI_Send(ownerOff.data(), static_cast<int>(ownerOff.size()), MPI_DOUBLE, 0, kOffdiagTag, ctx.comm));
    }
  } else if (ctx.rank == 0) {
    diag->resize(ctx.n);
    offdiag->resize(ctx.n - 1);
    MPI_CHECK(MPI_Recv(diag->data(), static_cast<int>(diag->size()), MPI_DOUBLE, owner, kDiagTag, ctx.comm, MPI_STATUS_IGNORE));
    MPI_CHECK(MPI_Recv(offdiag->data(), static_cast<int>(offdiag->size()), MPI_DOUBLE, owner, kOffdiagTag, ctx.comm, MPI_STATUS_IGNORE));
  }
}

