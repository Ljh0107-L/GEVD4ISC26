#pragma once

#include <cuda_runtime.h>

#include "runtime/error_handling.hpp"

#include <cstdio>
#include <optional>

#define EVD_CUDA_CHECK(call)                                                \
  do {                                                                      \
    const cudaError_t cuda_status_ = (call);                                \
    if (cuda_status_ != cudaSuccess) {                                      \
      std::fprintf(stderr,                                                  \
                   "CUDA error at %s:%d: %s (%d), call: %s\n",              \
                   __FILE__,                                                \
                   __LINE__,                                                \
                   cudaGetErrorString(cuda_status_),                        \
                   static_cast<int>(cuda_status_),                          \
                   #call);                                                  \
      ::gevd4isc26::evd::abortActiveCommunicator(                           \
          static_cast<int>(cuda_status_));                                  \
    }                                                                       \
  } while (0)

namespace gevd4isc26::evd {

class CudaEventTimer {
public:
  CudaEventTimer() {
    EVD_CUDA_CHECK(cudaEventCreate(&start_));
    EVD_CUDA_CHECK(cudaEventCreate(&stop_));
    EVD_CUDA_CHECK(cudaEventRecord(start_));
  }

  ~CudaEventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  CudaEventTimer(const CudaEventTimer&) = delete;
  CudaEventTimer& operator=(const CudaEventTimer&) = delete;

  [[nodiscard]] float stopMilliseconds() {
    EVD_CUDA_CHECK(cudaEventRecord(stop_));
    EVD_CUDA_CHECK(cudaEventSynchronize(stop_));
    float milliseconds = 0.0F;
    EVD_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
  }

private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

}  // namespace gevd4isc26::evd

// The source-derived kernels still use this small pair of timing helpers for
// opt-in fine-grained profiling. Keep their call sites stable while owning the
// CUDA events with RAII.
static thread_local std::optional<gevd4isc26::evd::CudaEventTimer>
    active_evd_timer;

static void startTimer() {
  active_evd_timer.reset();
  active_evd_timer.emplace();
}

static float stopTimer() {
  if (!active_evd_timer.has_value()) {
    return 0.0F;
  }
  const float milliseconds = active_evd_timer->stopMilliseconds();
  active_evd_timer.reset();
  return milliseconds;
}
