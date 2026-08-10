#pragma once

#include <cstddef>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <climits>
#include <sched.h>
#include <unistd.h>
#endif

namespace gevd4isc26::detail {

// Snapshot of the CPU mask granted to the current thread by the launcher or
// job scheduler. The mask is dynamically sized rather than using a fixed
// CPU_SETSIZE, so it remains valid on large and heterogeneous nodes.
class CpuAffinitySnapshot {
 public:
  CpuAffinitySnapshot() noexcept {
#if defined(__linux__)
    capture();
#endif
  }

  void restore() const noexcept {
#if defined(__linux__)
    restoreImpl();
#endif
  }

  CpuAffinitySnapshot(const CpuAffinitySnapshot&) = delete;
  CpuAffinitySnapshot& operator=(const CpuAffinitySnapshot&) = delete;
  CpuAffinitySnapshot(CpuAffinitySnapshot&&) = delete;
  CpuAffinitySnapshot& operator=(CpuAffinitySnapshot&&) = delete;

 private:
#if defined(__linux__)
  void capture() noexcept {
    try {
      const long configured_cpus = sysconf(_SC_NPROCESSORS_CONF);
      std::size_t bit_count =
          configured_cpus > 0 ? static_cast<std::size_t>(configured_cpus)
                              : static_cast<std::size_t>(CPU_SETSIZE);
      bit_count =
          bit_count < static_cast<std::size_t>(CPU_SETSIZE)
              ? static_cast<std::size_t>(CPU_SETSIZE)
              : bit_count;

      constexpr std::size_t kWordBits = sizeof(unsigned long) * CHAR_BIT;
      for (int attempt = 0; attempt < 8; ++attempt) {
        const std::size_t word_count =
            (bit_count + kWordBits - 1) / kWordBits;
        mask_.assign(word_count, 0UL);
        const std::size_t byte_count = word_count * sizeof(unsigned long);
        if (sched_getaffinity(
                0,
                byte_count,
                reinterpret_cast<cpu_set_t*>(mask_.data())) == 0) {
          byte_count_ = byte_count;
          captured_ = true;
          return;
        }
        if (errno != EINVAL) {
          break;
        }
        bit_count *= 2;
      }
    } catch (...) {
      // Affinity preservation is best-effort; numerical work remains valid if
      // the platform does not expose a usable affinity API or memory is tight.
    }
    mask_.clear();
    byte_count_ = 0;
    captured_ = false;
  }

  void restoreImpl() const noexcept {
    if (!captured_) {
      return;
    }
    const int saved_errno = errno;
    (void)sched_setaffinity(
        0,
        byte_count_,
        reinterpret_cast<const cpu_set_t*>(mask_.data()));
    errno = saved_errno;
  }

  std::vector<unsigned long> mask_;
  std::size_t byte_count_ = 0;
  bool captured_ = false;
#endif
};

// Several GPU communication initializers narrow their calling thread to the
// GPU-local NUMA node. Restoring only the caller lets later CPU work inherit
// the job's mask while leaving NCCL/NVSHMEM proxy threads on their
// vendor-selected CPUs.
class ScopedCpuAffinityRestore {
 public:
  ScopedCpuAffinityRestore() noexcept = default;

  ~ScopedCpuAffinityRestore() {
    snapshot_.restore();
  }

  ScopedCpuAffinityRestore(const ScopedCpuAffinityRestore&) = delete;
  ScopedCpuAffinityRestore& operator=(const ScopedCpuAffinityRestore&) =
      delete;
  ScopedCpuAffinityRestore(ScopedCpuAffinityRestore&&) = delete;
  ScopedCpuAffinityRestore& operator=(ScopedCpuAffinityRestore&&) = delete;

  void restoreNow() const noexcept {
    snapshot_.restore();
  }

 private:
  CpuAffinitySnapshot snapshot_;
};

}  // namespace gevd4isc26::detail
