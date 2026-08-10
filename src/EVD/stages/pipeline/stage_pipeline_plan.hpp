#pragma once

#include "engine/evd_types.hpp"

#include <algorithm>

namespace gevd4isc26::evd {

// A deliberately coarse SBR->BC pipeline.  Only rank zero leaves SBR early;
// it advances one dependency-safe batch of BC sweeps while every suffix rank
// finishes SBR.  Once the suffix is complete, the ordinary distributed BC
// range algorithm resumes from the saved frontier.  This keeps the overlap
// useful without turning the full BC reduction into thousands of tiny tasks.
struct StagePipelinePlan {
  bool enabled = false;
  long initialSweeps = 0;
  long segmentEnd = 0;
  long changedBandEnd = 0;
  long sbrGuardCols = 0;
  long sbrReleaseEnd = 0;
};

inline long alignUpTo(long value, long alignment)
{
  if (alignment <= 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

inline StagePipelinePlan buildStagePipelinePlan(const DistContext &ctx,
                                                 bool requested,
                                                 long configuredInitialSweeps = 0)
{
  StagePipelinePlan plan;
  if (!requested || ctx.size < 2 || ctx.counts.empty()) {
    return plan;
  }

  const long brownCols = std::max<long>(
      ctx.b,
      static_cast<long>(envIntOrDefault("EVD_BC_BROWN_COLS",
                                        static_cast<int>(4 * ctx.b))));
  const long rankZeroEnd = ctx.displs[0] + ctx.counts[0];
  const long totalSweeps = std::max<long>(0, ctx.n - 2);

  long initialSweeps = configuredInitialSweeps;
  if (initialSweeps <= 0) {
    initialSweeps =
        static_cast<long>(envIntOrDefault("EVD_STAGE_PIPELINE_SWEEPS", 0));
  }
  if (initialSweeps <= 0) {
    // One reasonably large cooperative launch is enough to overlap the SBR
    // suffix.  Scaling with rank zero's column block keeps the look-ahead
    // bounded on both small and large matrices.
    initialSweeps = std::max<long>(4, ctx.counts[0] / (4 * ctx.b));
    initialSweeps = std::min<long>(128, initialSweeps);
    initialSweeps = std::max<long>(4, ((initialSweeps + 3) / 4) * 4);
  }
  initialSweeps = std::min<long>(initialSweeps, totalSweeps);

  const long segmentEnd = std::min<long>(ctx.n, rankZeroEnd + brownCols);
  while (initialSweeps > 0) {
    const long slantedRows = 2 * ctx.b * (initialSweeps - 1);
    const long changedBandEnd =
        std::min<long>(ctx.n, segmentEnd + slantedRows + ctx.b);
    const long guardCols = brownCols + slantedRows + ctx.b;
    const long releaseEnd =
        std::min<long>(ctx.n, alignUpTo(rankZeroEnd + guardCols, ctx.nb));
    if (changedBandEnd < ctx.n && releaseEnd < ctx.n) {
      plan.enabled = true;
      plan.initialSweeps = initialSweeps;
      plan.segmentEnd = segmentEnd;
      plan.changedBandEnd = changedBandEnd;
      plan.sbrGuardCols = guardCols;
      plan.sbrReleaseEnd = releaseEnd;
      return plan;
    }
    initialSweeps /= 2;
  }
  return plan;
}

}  // namespace gevd4isc26::evd
