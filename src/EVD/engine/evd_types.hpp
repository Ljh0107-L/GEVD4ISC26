#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

namespace gevd4isc26::evd {

inline long ceilDiv(long a, long b)
{
  return (a + b - 1) / b;
}

inline bool isPowerOfTwo(long value)
{
  return value > 0 && (value & (value - 1)) == 0;
}

// The engine's column distribution, suffix communicators, and collective
// patterns are generic in the rank count; the power-of-two requirement keeps
// the SBR/BC sweep partitioning balanced.
inline bool isSupportedRankCount(int value)
{
  return isPowerOfTwo(value);
}

inline void buildColumnDistribution(long n,
                                    int ranks,
                                    std::vector<long> *counts,
                                    std::vector<long> *displs)
{
  counts->assign(ranks, n / ranks);
  for (int r = 0; r < static_cast<int>(n % ranks); ++r) {
    (*counts)[r]++;
  }
  displs->assign(ranks, 0);
  for (int r = 1; r < ranks; ++r) {
    (*displs)[r] = (*displs)[r - 1] + (*counts)[r - 1];
  }
}

inline void buildBlockAlignedColumnDistribution(long n,
                                                long blockSize,
                                                int ranks,
                                                std::vector<long> *counts,
                                                std::vector<long> *displs)
{
  const long totalBlocks = n / blockSize;
  counts->assign(ranks, 0);
  for (int r = 0; r < ranks; ++r) {
    const long blocks = totalBlocks / ranks +
                        ((r < static_cast<int>(totalBlocks % ranks)) ? 1 : 0);
    (*counts)[r] = blocks * blockSize;
  }
  displs->assign(ranks, 0);
  for (int r = 1; r < ranks; ++r) {
    (*displs)[r] = (*displs)[r - 1] + (*counts)[r - 1];
  }
}

inline void buildSweepDistributionFromColumnBlocks(long totalSweeps,
                                                   const std::vector<long> &columnCounts,
                                                   const std::vector<long> &columnDispls,
                                                   std::vector<long> *sweepCounts,
                                                   std::vector<long> *sweepDispls)
{
  const int ranks = static_cast<int>(columnCounts.size());
  sweepCounts->assign(ranks, 0);
  sweepDispls->assign(ranks, 0);
  for (int r = 0; r < ranks; ++r) {
    const long start = std::min(columnDispls[r], totalSweeps);
    const long end = std::min(columnDispls[r] + columnCounts[r], totalSweeps);
    (*sweepDispls)[r] = start;
    (*sweepCounts)[r] = std::max<long>(0, end - start);
  }
}

inline bool allCountsEqual(const std::vector<long> &counts)
{
  return std::all_of(counts.begin(), counts.end(), [&](long value) {
    return value == counts.front();
  });
}

inline int envIntOrDefault(const char *name, int defaultValue)
{
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return defaultValue;
  }
  return std::atoi(value);
}

inline double envDoubleOrDefault(const char *name, double defaultValue)
{
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return defaultValue;
  }
  return std::atof(value);
}

inline void buildBackTransformDistribution(long n,
                                           long blockSize,
                                           int ranks,
                                           int balancePct,
                                           bool alignToBlock,
                                           std::vector<long> *counts,
                                           std::vector<long> *displs)
{
  buildColumnDistribution(n, ranks, counts, displs);
  if (ranks <= 1 || balancePct <= 0) {
    return;
  }

  if (alignToBlock && blockSize > 0 && n % blockSize == 0 &&
      n / blockSize >= ranks) {
    const long totalBlocks = n / blockSize;
    const long baseBlocks = totalBlocks / ranks;
    const long maxDeltaBlocks = std::max<long>(1, baseBlocks * balancePct / 100);
    const double center = 0.5 * static_cast<double>(ranks - 1);
    std::vector<long> blockCounts(static_cast<size_t>(ranks), 0);
    long blockSum = 0;
    for (int r = 0; r < ranks; ++r) {
      const double scaled = (center - static_cast<double>(r)) / center;
      const long delta =
          static_cast<long>(std::llround(scaled * static_cast<double>(maxDeltaBlocks)));
      blockCounts[static_cast<size_t>(r)] = std::max<long>(1, baseBlocks + delta);
      blockSum += blockCounts[static_cast<size_t>(r)];
    }

    long diff = totalBlocks - blockSum;
    int rank = (diff >= 0) ? 0 : (ranks - 1);
    while (diff != 0) {
      if (diff > 0) {
        blockCounts[static_cast<size_t>(rank)]++;
        diff--;
        rank = (rank + 1) % ranks;
      } else {
        if (blockCounts[static_cast<size_t>(rank)] > 1) {
          blockCounts[static_cast<size_t>(rank)]--;
          diff++;
        }
        rank = (rank - 1 + ranks) % ranks;
      }
    }

    for (int r = 0; r < ranks; ++r) {
      (*counts)[r] = blockCounts[static_cast<size_t>(r)] * blockSize;
    }
    (*displs)[0] = 0;
    for (int r = 1; r < ranks; ++r) {
      (*displs)[r] = (*displs)[r - 1] + (*counts)[r - 1];
    }
    return;
  }

  const long base = n / ranks;
  const long maxDelta = std::max<long>(1, base * balancePct / 100);
  const double center = 0.5 * static_cast<double>(ranks - 1);
  long sum = 0;
  for (int r = 0; r < ranks; ++r) {
    const double scaled = (center - static_cast<double>(r)) / center;
    const long delta =
        static_cast<long>(std::llround(scaled * static_cast<double>(maxDelta)));
    (*counts)[r] = std::max<long>(1, base + delta);
    sum += (*counts)[r];
  }

  long diff = n - sum;
  int rank = 0;
  while (diff != 0) {
    const long step = (diff > 0) ? 1 : -1;
    if ((*counts)[rank] + step > 0) {
      (*counts)[rank] += step;
      diff -= step;
    }
    rank = (rank + 1) % ranks;
  }

  (*displs)[0] = 0;
  for (int r = 1; r < ranks; ++r) {
    (*displs)[r] = (*displs)[r - 1] + (*counts)[r - 1];
  }
}

inline int ownerOfColumn(long col,
                         const std::vector<long> &counts,
                         const std::vector<long> &displs)
{
  for (int r = 0; r < static_cast<int>(counts.size()); ++r) {
    if (col >= displs[r] && col < displs[r] + counts[r]) {
      return r;
    }
  }
  return static_cast<int>(counts.size()) - 1;
}

struct DistContext {
  int rank = 0;
  int size = 1;
  long n = 0;
  long b = 0;
  long nb = 0;
  long colStart = 0;
  long localCols = 0;
  std::vector<long> counts;
  std::vector<long> displs;
  long qColStart = 0;
  long qLocalCols = 0;
  std::vector<long> qCounts;
  std::vector<long> qDispls;
  MPI_Comm comm = MPI_COMM_WORLD;
  ncclComm_t nccl = nullptr;
  std::vector<ncclComm_t> sbrSuffixNccls;
  cudaStream_t commStream = nullptr;
};

struct SbrPipelineEarlyReturnInfo {
  int activeRanks = 0;
  int firstRank = -1;
  long firstEarlyEnd = 0;
};

inline SbrPipelineEarlyReturnInfo computeSbrPipelineEarlyReturnInfo(const DistContext &ctx,
                                                                    long guardCols)
{
  SbrPipelineEarlyReturnInfo info;
  info.firstEarlyEnd = ctx.n;
  for (int r = 0; r < ctx.size; ++r) {
    const long earlyEnd = ctx.displs[static_cast<size_t>(r)] +
                          ctx.counts[static_cast<size_t>(r)] + guardCols;
    if (earlyEnd < ctx.n) {
      info.activeRanks++;
      if (info.firstRank < 0) {
        info.firstRank = r;
        info.firstEarlyEnd = earlyEnd;
      }
    }
  }
  return info;
}

struct SbrWorkspace {
  double *dWork = nullptr;
  double *dR = nullptr;
  double *dWPanel = nullptr;
  double *dYPanel = nullptr;
  double *dWBT = nullptr;
  double *dYBT = nullptr;
  double *dAWBT = nullptr;
  double *dZBT = nullptr;
  double *dZFullBT = nullptr;
  double *dSLocal = nullptr;
  double *dS = nullptr;
  double *dABase = nullptr;
  double *dYBlock = nullptr;
  double *dYBlockBT = nullptr;
  double *dZBlockBT = nullptr;
  double *dTmpPrev1 = nullptr;
  double *dTmpPrev2 = nullptr;
};

struct SbrStats {
  unsigned long long panels = 0;
  unsigned long long activePanels = 0;
  unsigned long long wyBroadcastDoubles = 0;
  unsigned long long zBroadcastDoubles = 0;
  unsigned long long allReduceDoubles = 0;
  unsigned long long updateGemmCalls = 0;
  unsigned long long tailSyr2kCalls = 0;
  unsigned long long skippedZBroadcastPanels = 0;
  unsigned long long localUpdateEntries = 0;
};

}  // namespace gevd4isc26::evd
