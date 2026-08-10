#pragma once

#include "runtime/error.hpp"

#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>

namespace gevd4isc26::detail {

class DeviceAllocation {
public:
  DeviceAllocation() = default;
  explicit DeviceAllocation(std::size_t bytes) { allocate(bytes); }
  ~DeviceAllocation() { reset(); }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  DeviceAllocation(DeviceAllocation&& other) noexcept { swap(other); }
  DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
    if (this != &other) {
      reset();
      swap(other);
    }
    return *this;
  }

  void allocate(std::size_t bytes) {
    reset();
    if (bytes != 0) {
      GEVD_CUDA(cudaMalloc(&pointer_, bytes));
    }
    bytes_ = bytes;
  }

  void reset() noexcept {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
    pointer_ = nullptr;
    bytes_ = 0;
  }

  template <class T>
  [[nodiscard]] T* as() noexcept {
    return static_cast<T*>(pointer_);
  }

  template <class T>
  [[nodiscard]] const T* as() const noexcept {
    return static_cast<const T*>(pointer_);
  }

  [[nodiscard]] void* data() noexcept { return pointer_; }
  [[nodiscard]] const void* data() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
  void swap(DeviceAllocation& other) noexcept {
    std::swap(pointer_, other.pointer_);
    std::swap(bytes_, other.bytes_);
  }

  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

class HostAllocation {
public:
  HostAllocation() = default;
  explicit HostAllocation(std::size_t bytes) { allocate(bytes); }
  ~HostAllocation() { reset(); }

  HostAllocation(const HostAllocation&) = delete;
  HostAllocation& operator=(const HostAllocation&) = delete;

  HostAllocation(HostAllocation&& other) noexcept { swap(other); }
  HostAllocation& operator=(HostAllocation&& other) noexcept {
    if (this != &other) {
      reset();
      swap(other);
    }
    return *this;
  }

  void allocate(std::size_t bytes) {
    reset();
    if (bytes != 0) {
      pointer_ = std::malloc(bytes);
      if (pointer_ == nullptr) {
        throw std::bad_alloc();
      }
    }
    bytes_ = bytes;
  }

  void reset() noexcept {
    std::free(pointer_);
    pointer_ = nullptr;
    bytes_ = 0;
  }

  [[nodiscard]] void* data() noexcept { return pointer_; }
  [[nodiscard]] const void* data() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

private:
  void swap(HostAllocation& other) noexcept {
    std::swap(pointer_, other.pointer_);
    std::swap(bytes_, other.bytes_);
  }

  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

}  // namespace gevd4isc26::detail
