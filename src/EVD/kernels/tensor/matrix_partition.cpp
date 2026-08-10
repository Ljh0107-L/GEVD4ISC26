#include "matrix_partition.hpp"

#include <algorithm>

std::vector<std::int64_t> partitionSymmetricRank2k(std::int64_t order) {
  std::vector<std::int64_t> partitions;
  while (order > 0) {
    std::int64_t partition = 0;
    if (order < 8192) {
      partition = order;
    } else {
      partition = std::min<std::int64_t>(order, 16384);
      if (partition < 16384) {
        partition = 8192;
      }
    }
    partitions.push_back(partition);
    order -= partition;
  }
  return partitions;
}
