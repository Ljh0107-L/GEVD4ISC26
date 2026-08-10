#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace gevd4isc26::evd::d_and_c {

inline constexpr int kInvalidArgument = -1000;
inline constexpr int kAllocationFailure = -1010;
inline constexpr int kThreadLaunchFailure = -1011;

struct SolveResult {
  int info = kInvalidArgument;
  double milliseconds = 0.0;
};

// Returns whether DSTEDC's documented O(n^2) workspace no longer fits in a
// 32-bit workspace-length argument.
bool needs64BitWorkspace(std::int64_t order) noexcept;

// CPU-only synchronous primitive. diagonal and off_diagonal are overwritten
// by LAPACK; eigenvectors is an order-by-order column-major output matrix.
SolveResult solveTridiagonal(std::int64_t order,
                             double* diagonal,
                             double* off_diagonal,
                             double* eigenvectors) noexcept;

// Owns an asynchronous CPU D&C solve and its large host eigenvector buffer.
// Destruction waits for an outstanding solve, which makes early returns in a
// caller safe. Instances are one-shot and movable, but not copyable.
class AsyncTridiagonalSolve {
 public:
  AsyncTridiagonalSolve() noexcept;
  ~AsyncTridiagonalSolve();

  AsyncTridiagonalSolve(const AsyncTridiagonalSolve&) = delete;
  AsyncTridiagonalSolve& operator=(const AsyncTridiagonalSolve&) = delete;
  AsyncTridiagonalSolve(AsyncTridiagonalSolve&&) noexcept;
  AsyncTridiagonalSolve& operator=(AsyncTridiagonalSolve&&) noexcept;

  int start(std::int64_t order,
            std::vector<double> diagonal,
            std::vector<double> off_diagonal) noexcept;
  int wait() noexcept;

  const std::vector<double>& eigenvalues() const noexcept;
  const double* eigenvectors() const noexcept;
  std::int64_t order() const noexcept;
  double solveMilliseconds() const noexcept;
  bool started() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gevd4isc26::evd::d_and_c
