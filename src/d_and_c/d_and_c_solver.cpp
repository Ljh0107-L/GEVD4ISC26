#include "d_and_c_solver.hpp"

#include <mkl_lapack.h>
#include <mkl_lapacke.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <new>
#include <thread>
#include <utility>

namespace gevd4isc26::evd::d_and_c {
namespace {

int environmentInteger(const char* name, int fallback) noexcept {
  const char* text = std::getenv(name);
  if (text == nullptr || *text == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    return fallback;
  }
  return static_cast<int>(value);
}

bool hostEigenvectorCacheEnabled() noexcept {
  return environmentInteger("EVD_HOST_Z_CACHE", 1) != 0;
}

void firstTouch(double* pointer, std::size_t elements) {
  if (pointer == nullptr || elements == 0) {
    return;
  }

  int thread_count = environmentInteger(
      "EVD_HOST_Z_TOUCH_THREADS",
      environmentInteger("MKL_NUM_THREADS", 64));
  const unsigned int hardware_threads = std::thread::hardware_concurrency();
  if (hardware_threads > 0) {
    thread_count =
        std::min(thread_count, static_cast<int>(hardware_threads));
  }
  thread_count = std::max(1, thread_count);
  thread_count = std::min(
      thread_count,
      static_cast<int>(std::min<std::size_t>(elements, 4096)));

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(thread_count));
  try {
    for (int thread = 0; thread < thread_count; ++thread) {
      const std::size_t begin =
          elements * static_cast<std::size_t>(thread) /
          static_cast<std::size_t>(thread_count);
      const std::size_t end =
          elements * static_cast<std::size_t>(thread + 1) /
          static_cast<std::size_t>(thread_count);
      workers.emplace_back([pointer, begin, end]() {
        std::fill(pointer + begin, pointer + end, 0.0);
      });
    }
  } catch (...) {
    for (std::thread& worker : workers) {
      worker.join();
    }
    throw;
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
}

struct HostEigenvectorCache {
  double* pointer = nullptr;
  std::size_t elements = 0;
  bool in_use = false;
  std::mutex mutex;

  ~HostEigenvectorCache() {
    std::free(pointer);
  }
};

HostEigenvectorCache& hostEigenvectorCache() {
  static HostEigenvectorCache cache;
  return cache;
}

struct HostBuffer {
  double* pointer = nullptr;
  bool cached = false;
};

HostBuffer acquireHostBuffer(std::size_t elements) {
  if (!hostEigenvectorCacheEnabled()) {
    return {
        static_cast<double*>(std::malloc(sizeof(double) * elements)),
        false,
    };
  }

  HostEigenvectorCache& cache = hostEigenvectorCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  if (cache.in_use) {
    return {
        static_cast<double*>(std::malloc(sizeof(double) * elements)),
        false,
    };
  }
  if (cache.pointer == nullptr || cache.elements < elements) {
    std::free(cache.pointer);
    cache.pointer =
        static_cast<double*>(std::malloc(sizeof(double) * elements));
    cache.elements = 0;
    if (cache.pointer == nullptr) {
      return {};
    }
    cache.elements = elements;
    firstTouch(cache.pointer, elements);
  }
  cache.in_use = true;
  return {cache.pointer, true};
}

void releaseHostBuffer(HostBuffer buffer) noexcept {
  if (buffer.pointer == nullptr) {
    return;
  }
  if (!buffer.cached) {
    std::free(buffer.pointer);
    return;
  }

  HostEigenvectorCache& cache = hostEigenvectorCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  if (cache.pointer == buffer.pointer) {
    cache.in_use = false;
  }
}

bool validOutputSize(std::int64_t order, std::size_t* elements) noexcept {
  if (order <= 0) {
    return false;
  }
  const auto unsigned_order = static_cast<std::uint64_t>(order);
  if (unsigned_order >
      std::numeric_limits<std::size_t>::max() / unsigned_order) {
    return false;
  }
  const std::size_t count =
      static_cast<std::size_t>(unsigned_order * unsigned_order);
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    return false;
  }
  *elements = count;
  return true;
}

}  // namespace

bool needs64BitWorkspace(std::int64_t order) noexcept {
  if (order <= 0) {
    return false;
  }
  const auto unsigned_order = static_cast<std::uint64_t>(order);
  const auto maximum_32_bit = static_cast<std::uint64_t>(INT_MAX);
  if (unsigned_order > maximum_32_bit) {
    return true;
  }
  return unsigned_order * unsigned_order + 4 * unsigned_order + 1 >
         maximum_32_bit;
}

SolveResult solveTridiagonal(std::int64_t order,
                             double* diagonal,
                             double* off_diagonal,
                             double* eigenvectors) noexcept {
  SolveResult result;
  if (order <= 0 || diagonal == nullptr || eigenvectors == nullptr ||
      (order > 1 && off_diagonal == nullptr)) {
    return result;
  }

  const auto start = std::chrono::steady_clock::now();
  try {
    if (needs64BitWorkspace(order)) {
      const MKL_INT64 n = static_cast<MKL_INT64>(order);
      const MKL_INT64 leading_dimension = n;
      const MKL_INT64 work_size = 1 + 4 * n + n * n;
      const MKL_INT64 integer_work_size = 3 + 5 * n;
      std::vector<double> work(static_cast<std::size_t>(work_size));
      std::vector<MKL_INT64> integer_work(
          static_cast<std::size_t>(integer_work_size));
      MKL_INT64 info = 0;
      const char compute_eigenvectors = 'I';
      dstedc_64(&compute_eigenvectors,
                &n,
                diagonal,
                off_diagonal,
                eigenvectors,
                &leading_dimension,
                work.data(),
                &work_size,
                integer_work.data(),
                &integer_work_size,
                &info);
      result.info = static_cast<int>(info);
    } else {
      result.info = static_cast<int>(
          LAPACKE_dstedc(LAPACK_COL_MAJOR,
                         'I',
                         static_cast<lapack_int>(order),
                         diagonal,
                         off_diagonal,
                         eigenvectors,
                         static_cast<lapack_int>(order)));
    }
  } catch (const std::bad_alloc&) {
    result.info = kAllocationFailure;
  } catch (...) {
    result.info = kInvalidArgument;
  }
  const auto stop = std::chrono::steady_clock::now();
  result.milliseconds =
      std::chrono::duration<double, std::milli>(stop - start).count();
  return result;
}

struct AsyncTridiagonalSolve::Impl {
  std::int64_t order = 0;
  std::vector<double> diagonal;
  std::vector<double> off_diagonal;
  HostBuffer eigenvector_buffer;
  std::thread worker;
  int info = kInvalidArgument;
  double milliseconds = 0.0;
  bool launched = false;
  bool joined = false;

  ~Impl() {
    if (worker.joinable()) {
      worker.join();
    }
    releaseHostBuffer(eigenvector_buffer);
  }
};

AsyncTridiagonalSolve::AsyncTridiagonalSolve() noexcept = default;

AsyncTridiagonalSolve::~AsyncTridiagonalSolve() = default;

AsyncTridiagonalSolve::AsyncTridiagonalSolve(
    AsyncTridiagonalSolve&&) noexcept = default;

AsyncTridiagonalSolve& AsyncTridiagonalSolve::operator=(
    AsyncTridiagonalSolve&&) noexcept = default;

int AsyncTridiagonalSolve::start(
    std::int64_t order,
    std::vector<double> diagonal,
    std::vector<double> off_diagonal) noexcept {
  if (impl_ != nullptr || order <= 0 ||
      diagonal.size() != static_cast<std::size_t>(order) ||
      off_diagonal.size() != static_cast<std::size_t>(order - 1)) {
    return kInvalidArgument;
  }

  std::size_t eigenvector_elements = 0;
  if (!validOutputSize(order, &eigenvector_elements)) {
    return kAllocationFailure;
  }

  std::unique_ptr<Impl> state(new (std::nothrow) Impl);
  if (state == nullptr) {
    return kAllocationFailure;
  }
  state->order = order;
  state->diagonal = std::move(diagonal);
  state->off_diagonal = std::move(off_diagonal);
  try {
    state->eigenvector_buffer = acquireHostBuffer(eigenvector_elements);
  } catch (const std::bad_alloc&) {
    return kAllocationFailure;
  } catch (...) {
    return kAllocationFailure;
  }
  if (state->eigenvector_buffer.pointer == nullptr) {
    return kAllocationFailure;
  }

  Impl* raw_state = state.get();
  try {
    state->worker = std::thread([raw_state]() {
      const SolveResult result =
          solveTridiagonal(raw_state->order,
                           raw_state->diagonal.data(),
                           raw_state->off_diagonal.data(),
                           raw_state->eigenvector_buffer.pointer);
      raw_state->info = result.info;
      raw_state->milliseconds = result.milliseconds;
    });
  } catch (...) {
    return kThreadLaunchFailure;
  }
  state->launched = true;
  impl_ = std::move(state);
  return 0;
}

int AsyncTridiagonalSolve::wait() noexcept {
  if (impl_ == nullptr || !impl_->launched) {
    return kInvalidArgument;
  }
  if (!impl_->joined && impl_->worker.joinable()) {
    impl_->worker.join();
    impl_->joined = true;
  }
  return impl_->info;
}

const std::vector<double>& AsyncTridiagonalSolve::eigenvalues() const noexcept {
  static const std::vector<double> empty;
  return impl_ == nullptr ? empty : impl_->diagonal;
}

const double* AsyncTridiagonalSolve::eigenvectors() const noexcept {
  return impl_ == nullptr ? nullptr : impl_->eigenvector_buffer.pointer;
}

std::int64_t AsyncTridiagonalSolve::order() const noexcept {
  return impl_ == nullptr ? 0 : impl_->order;
}

double AsyncTridiagonalSolve::solveMilliseconds() const noexcept {
  return impl_ == nullptr ? 0.0 : impl_->milliseconds;
}

bool AsyncTridiagonalSolve::started() const noexcept {
  return impl_ != nullptr && impl_->launched;
}

}  // namespace gevd4isc26::evd::d_and_c
