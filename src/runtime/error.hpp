#pragma once

#include <cuda_runtime.h>
#include <cublasmp.h>
#include <cusolverMp.h>
#include <mpi.h>
#include <nccl.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace gevd4isc26::detail {

inline std::string location(const char* expression, const char* file, int line) {
  std::ostringstream message;
  message << expression << " at " << file << ':' << line;
  return message.str();
}

inline void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(location(expression, file, line) + " failed: " +
                             cudaGetErrorString(status));
  }
}

inline void checkNccl(ncclResult_t status, const char* expression, const char* file, int line) {
  if (status != ncclSuccess) {
    throw std::runtime_error(location(expression, file, line) + " failed: " +
                             ncclGetErrorString(status));
  }
}

inline void checkCusolver(cusolverStatus_t status,
                          const char* expression,
                          const char* file,
                          int line) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(location(expression, file, line) +
                             " failed with cuSOLVER status " +
                             std::to_string(static_cast<int>(status)));
  }
}

inline void checkCublasMp(cublasMpStatus_t status,
                          const char* expression,
                          const char* file,
                          int line) {
  if (status != CUBLASMP_STATUS_SUCCESS) {
    throw std::runtime_error(location(expression, file, line) +
                             " failed with cuBLASMp status " +
                             std::to_string(static_cast<int>(status)));
  }
}

inline void checkMpi(int status, const char* expression, const char* file, int line) {
  if (status == MPI_SUCCESS) {
    return;
  }
  char error_text[MPI_MAX_ERROR_STRING] = {};
  int length = 0;
  MPI_Error_string(status, error_text, &length);
  throw std::runtime_error(location(expression, file, line) + " failed: " +
                           std::string(error_text, length));
}

}  // namespace gevd4isc26::detail

#define GEVD_CUDA(expression) \
  ::gevd4isc26::detail::checkCuda((expression), #expression, __FILE__, __LINE__)
#define GEVD_NCCL(expression) \
  ::gevd4isc26::detail::checkNccl((expression), #expression, __FILE__, __LINE__)
#define GEVD_CUSOLVER(expression) \
  ::gevd4isc26::detail::checkCusolver((expression), #expression, __FILE__, __LINE__)
#define GEVD_CUBLASMP(expression) \
  ::gevd4isc26::detail::checkCublasMp((expression), #expression, __FILE__, __LINE__)
#define GEVD_MPI(expression) \
  ::gevd4isc26::detail::checkMpi((expression), #expression, __FILE__, __LINE__)
