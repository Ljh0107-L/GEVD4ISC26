#pragma once

#include <cstdint>
#include <vector>

// Splits a symmetric rank-2k update into the same 16384/8192/remainder
// sequence used by the source-derived tensor kernels. Returning a vector keeps
// ownership local to the caller; the former raw allocation was never freed.
[[nodiscard]] std::vector<std::int64_t>
partitionSymmetricRank2k(std::int64_t order);
