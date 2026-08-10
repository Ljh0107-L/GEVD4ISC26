constexpr int kBcBandwidth = 32;
constexpr long kBcBackPackedUColumnTile =
    ::gevd4isc26::evd::kBcBackColumnTile;

long computeSbrPipelineGuardCols(long n, long b, long nb)
{
  long brownCols =
      static_cast<long>(envIntOrDefault("EVD_BC_BROWN_COLS", static_cast<int>(4 * b)));
  brownCols = std::max<long>(0, brownCols);

  long guardCols = brownCols;
  const bool rollingBc = envIntOrDefault("EVD_BC_SEGMENTED_ROLLING", 0) != 0;
  const bool tileMultiStep = envIntOrDefault("EVD_BC_SEGMENTED_TILE_MULTI_STEP", 0) != 0;
  if (rollingBc || tileMultiStep) {
    long batchSweeps =
        static_cast<long>(envIntOrDefault("EVD_BC_SEGMENTED_BATCH_SWEEPS", 0));
    if (batchSweeps <= 0) {
      batchSweeps = std::max<long>(1, brownCols / (2 * b) + 1);
    }
    guardCols += 2 * b * std::max<long>(0, batchSweeps - 1);
    guardCols += std::max<long>(0, nb);
  }

  const char *explicitGuard = std::getenv("EVD_SBR_PIPELINE_GUARD_COLS");
  if (explicitGuard != nullptr && explicitGuard[0] != '\0') {
    guardCols = std::atol(explicitGuard);
  }
  return std::max<long>(0, std::min<long>(n, guardCols));
}
