#include "gevd/factor_cache.hpp"

#include "runtime/error.hpp"

#include <algorithm>
#include <cstring>

namespace gevd4isc26::detail {
namespace {

std::uint64_t sampledFingerprint(const DistributedMatrixView& matrix) {
  std::uint64_t hash = 1469598103934665603ull;
  auto mix = [&hash](std::uint64_t word) {
    hash ^= word;
    hash *= 1099511628211ull;
  };

  const auto& distribution = matrix.distribution;
  mix(static_cast<std::uint64_t>(distribution.order()));
  mix(static_cast<std::uint64_t>(matrix.leading_dimension));
  mix(static_cast<std::uint64_t>(distribution.localColumns()));

  const std::int64_t elements =
      matrix.leading_dimension * distribution.localColumns();
  constexpr std::int64_t sample_limit = 4096;
  const std::int64_t samples = std::min(sample_limit, elements);
  for (std::int64_t sample = 0; sample < samples; ++sample) {
    const std::int64_t index =
        samples > 1 ? sample * (elements - 1) / (samples - 1) : 0;
    std::uint64_t word = 0;
    std::memcpy(&word, matrix.data + index, sizeof(word));
    mix(static_cast<std::uint64_t>(index));
    mix(word);
  }

  // This mirrors the additional local-diagonal sampling used by the DFTB+
  // implementation. The main evenly-spaced samples already cover every
  // distribution, while these samples make changes near the diagonal cheap
  // to detect for repeatedly solved problems.
  const std::int64_t diagonal_count =
      std::min({distribution.order(), distribution.localRows(),
                distribution.localColumns()});
  const std::int64_t diagonal_samples =
      std::min<std::int64_t>(1024, diagonal_count);
  for (std::int64_t sample = 0; sample < diagonal_samples; ++sample) {
    const std::int64_t row = diagonal_samples > 1
                                 ? sample * (diagonal_count - 1) /
                                       (diagonal_samples - 1)
                                 : 0;
    const std::int64_t index = row + row * matrix.leading_dimension;
    std::uint64_t word = 0;
    std::memcpy(&word, matrix.data + index, sizeof(word));
    mix(static_cast<std::uint64_t>(index));
    mix(word);
  }
  return hash;
}

bool allRanksTrue(const DistributedGpuContext& context, bool local_value) {
  int local = local_value ? 1 : 0;
  int global = 0;
  GEVD_MPI(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN,
                         context.communicator()));
  return global != 0;
}

}  // namespace

FactorReuse FactorizationCache::prepare(
    const DistributedGpuContext& context,
    const DistributedMatrixView& B,
    std::int64_t inverse_leading_dimension,
    bool cache_cholesky,
    bool cache_inverse,
    bool need_inverse) {
  const auto& distribution = B.distribution;
  const std::size_t cholesky_bytes =
      distribution.storageSize(B.leading_dimension) * sizeof(double);
  const std::size_t inverse_bytes =
      distribution.storageSize(inverse_leading_dimension) * sizeof(double);
  const ProcessGrid& grid = distribution.grid();
  const MatrixKey current_key{
      distribution.order(),
      distribution.rowBlockSize(),
      distribution.columnBlockSize(),
      B.leading_dimension,
      distribution.localColumns(),
      grid.rows,
      grid.columns,
      grid.source_row,
      grid.source_column,
      distribution.processRow(),
      distribution.processColumn(),
      cholesky_bytes,
      cache_cholesky ? sampledFingerprint(B) : 0};

  const bool local_cholesky_reuse =
      cache_cholesky && cholesky_valid_ && cholesky_key_set_ &&
      cholesky_key_ == current_key;
  const bool reuse_cholesky =
      allRanksTrue(context, local_cholesky_reuse);

  if (!reuse_cholesky) {
    if (cholesky_.bytes() != cholesky_bytes) {
      cholesky_.allocate(cholesky_bytes);
    }
    cholesky_key_ = current_key;
    cholesky_key_set_ = true;
    cholesky_valid_ = false;
    inverse_valid_ = false;
  }

  bool reuse_inverse = false;
  if (need_inverse) {
    const bool local_inverse_reuse =
        cache_inverse && cache_cholesky && reuse_cholesky &&
        inverse_valid_ && inverse_key_set_ &&
        inverse_basis_key_ == current_key &&
        inverse_leading_dimension_ == inverse_leading_dimension &&
        inverse_bytes_ == inverse_bytes;
    reuse_inverse = allRanksTrue(context, local_inverse_reuse);
    if (!reuse_inverse) {
      if (inverse_.bytes() != inverse_bytes) {
        inverse_.allocate(inverse_bytes);
      }
      inverse_basis_key_ = current_key;
      inverse_leading_dimension_ = inverse_leading_dimension;
      inverse_bytes_ = inverse_bytes;
      inverse_key_set_ = true;
      inverse_valid_ = false;
    }
  }

  return FactorReuse{reuse_cholesky, reuse_inverse};
}

void FactorizationCache::markCholeskyValid(bool cache_enabled) noexcept {
  cholesky_valid_ = cache_enabled;
  if (!cache_enabled) {
    inverse_valid_ = false;
  }
}

void FactorizationCache::markInverseValid(bool cache_enabled) noexcept {
  inverse_valid_ = cache_enabled && cholesky_valid_;
}

}  // namespace gevd4isc26::detail
