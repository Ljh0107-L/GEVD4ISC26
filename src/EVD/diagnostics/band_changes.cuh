void printChangedBandColumns(const std::string &label,
                             long n,
                             long b,
                             double *dBefore,
                             double *dAfter,
                             double eps)
{
  const long ldBand = 2 * b;
  const size_t bandElems = static_cast<size_t>(ldBand) * static_cast<size_t>(n);
  std::vector<double> before(bandElems);
  std::vector<double> after(bandElems);
  EVD_CUDA_CHECK(cudaMemcpy(before.data(), dBefore, sizeof(double) * bandElems, cudaMemcpyDeviceToHost));
  EVD_CUDA_CHECK(cudaMemcpy(after.data(), dAfter, sizeof(double) * bandElems, cudaMemcpyDeviceToHost));
  long firstChangedCol = n;
  long lastChangedCol = -1;
  long maxDiffCol = -1;
  long maxDiffRow = -1;
  double maxDiff = 0.0;
  for (long col = 0; col < n; ++col) {
    bool colChanged = false;
    for (long row = 0; row < ldBand; ++row) {
      const size_t idx = static_cast<size_t>(col) * static_cast<size_t>(ldBand) +
                         static_cast<size_t>(row);
      const double diff = std::abs(after[idx] - before[idx]);
      if (diff > eps) {
        colChanged = true;
      }
      if (diff > maxDiff) {
        maxDiff = diff;
        maxDiffCol = col;
        maxDiffRow = row;
      }
    }
    if (colChanged) {
      firstChangedCol = std::min(firstChangedCol, col);
      lastChangedCol = std::max(lastChangedCol, col);
    }
  }
  if (lastChangedCol >= firstChangedCol) {
    std::cout << label
              << " first=" << firstChangedCol
              << " last=" << lastChangedCol
              << " count=" << (lastChangedCol - firstChangedCol + 1)
              << " max_diff=" << std::setprecision(17) << maxDiff
              << " max_col=" << maxDiffCol
              << " max_row_offset=" << maxDiffRow
              << " eps=" << eps << std::endl;
  } else {
    std::cout << label << " none eps=" << eps << std::endl;
  }
}
