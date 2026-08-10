#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <mpi.h>
#include <nccl.h>

#include <iostream>
#include <string>

namespace gevd4isc26::evd {

[[nodiscard]] MPI_Comm activeErrorCommunicator() noexcept;
MPI_Comm setActiveErrorCommunicator(MPI_Comm communicator) noexcept;
[[noreturn]] void abortActiveCommunicator(int code) noexcept;

class ErrorCommunicatorScope {
public:
  explicit ErrorCommunicatorScope(MPI_Comm communicator)
      : previous_(setActiveErrorCommunicator(communicator)) {}

  ~ErrorCommunicatorScope() {
    setActiveErrorCommunicator(previous_);
  }

  ErrorCommunicatorScope(const ErrorCommunicatorScope&) = delete;
  ErrorCommunicatorScope& operator=(const ErrorCommunicatorScope&) = delete;

private:
  MPI_Comm previous_;
};

}  // namespace gevd4isc26::evd

#define MPI_CHECK(call)                                                       \
  do {                                                                        \
    const int mpi_status_ = (call);                                           \
    if (mpi_status_ != MPI_SUCCESS) {                                         \
      char error_text_[MPI_MAX_ERROR_STRING] = {};                            \
      int error_length_ = 0;                                                  \
      MPI_Error_string(mpi_status_, error_text_, &error_length_);             \
      std::cerr << "MPI error at " << __FILE__ << ':' << __LINE__ << ": "    \
                << std::string(error_text_, error_length_) << std::endl;       \
      ::gevd4isc26::evd::abortActiveCommunicator(mpi_status_);                \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK_LOCAL(call)                                              \
  do {                                                                        \
    const cublasStatus_t status_ = (call);                                    \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                   \
      std::cerr << "cuBLAS error at " << __FILE__ << ':' << __LINE__ << ": " \
                << static_cast<int>(status_) << std::endl;                    \
      ::gevd4isc26::evd::abortActiveCommunicator(1);                          \
    }                                                                         \
  } while (0)

#define CUSOLVER_CHECK_LOCAL(call)                                            \
  do {                                                                        \
    const cusolverStatus_t status_ = (call);                                  \
    if (status_ != CUSOLVER_STATUS_SUCCESS) {                                 \
      std::cerr << "cuSOLVER error at " << __FILE__ << ':' << __LINE__       \
                << ": " << static_cast<int>(status_) << std::endl;            \
      ::gevd4isc26::evd::abortActiveCommunicator(1);                          \
    }                                                                         \
  } while (0)

#define NCCL_CHECK_LOCAL(call)                                                \
  do {                                                                        \
    const ncclResult_t status_ = (call);                                      \
    if (status_ != ncclSuccess) {                                             \
      std::cerr << "NCCL error at " << __FILE__ << ':' << __LINE__ << ": "   \
                << ncclGetErrorString(status_) << std::endl;                  \
      ::gevd4isc26::evd::abortActiveCommunicator(1);                          \
    }                                                                         \
  } while (0)
