void dumpTridiagonalIfRequested(long n, const std::vector<double> &diag, const std::vector<double> &offdiag)
{
  const char *path = std::getenv("EVD_DUMP_TRIDIAG");
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::cerr << "failed to open EVD_DUMP_TRIDIAG path: " << path << std::endl;
    return;
  }
  const std::int64_t n64 = static_cast<std::int64_t>(n);
  out.write(reinterpret_cast<const char *>(&n64), sizeof(n64));
  out.write(reinterpret_cast<const char *>(diag.data()), sizeof(double) * diag.size());
  out.write(reinterpret_cast<const char *>(offdiag.data()), sizeof(double) * offdiag.size());
  if (!out) {
    std::cerr << "failed to write EVD_DUMP_TRIDIAG path: " << path << std::endl;
    return;
  }
  std::cout << "EVD_DUMP_TRIDIAG wrote " << path << " n=" << n << std::endl;
}
