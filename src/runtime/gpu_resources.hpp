#pragma once

#include <cuda_runtime.h>
#include <cublasmp.h>
#include <cusolverMp.h>
#include <mpi.h>
#include <nccl.h>

namespace gevd4isc26::detail {

class CudaStream {
public:
  CudaStream() = default;
  ~CudaStream();

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  void create(int device);
  void reset() noexcept;

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
  int device_ = -1;
  cudaStream_t stream_ = nullptr;
};

class NcclCommunicator {
public:
  NcclCommunicator() = default;
  ~NcclCommunicator();

  NcclCommunicator(const NcclCommunicator&) = delete;
  NcclCommunicator& operator=(const NcclCommunicator&) = delete;

  void create(MPI_Comm communicator, int rank, int size);
  void reset() noexcept;

  [[nodiscard]] ncclComm_t get() const noexcept { return communicator_; }

private:
  ncclComm_t communicator_ = nullptr;
};

class SolverMpResources {
public:
  SolverMpResources() = default;
  ~SolverMpResources();

  SolverMpResources(const SolverMpResources&) = delete;
  SolverMpResources& operator=(const SolverMpResources&) = delete;

  void create(int device,
              cudaStream_t stream,
              ncclComm_t nccl,
              int grid_rows,
              int grid_columns);
  void reset() noexcept;

  [[nodiscard]] cusolverMpHandle_t handle() const noexcept { return handle_; }
  [[nodiscard]] cusolverMpGrid_t grid() const noexcept { return grid_; }

private:
  cusolverMpHandle_t handle_ = nullptr;
  cusolverMpGrid_t grid_ = nullptr;
};

class BlasMpResources {
public:
  BlasMpResources() = default;
  ~BlasMpResources();

  BlasMpResources(const BlasMpResources&) = delete;
  BlasMpResources& operator=(const BlasMpResources&) = delete;

  void create(cudaStream_t stream,
              ncclComm_t nccl,
              int grid_rows,
              int grid_columns);
  void reset() noexcept;

  [[nodiscard]] cublasMpHandle_t handle() const noexcept { return handle_; }
  [[nodiscard]] cublasMpGrid_t grid() const noexcept { return grid_; }

private:
  cublasMpHandle_t handle_ = nullptr;
  cublasMpGrid_t grid_ = nullptr;
};

}  // namespace gevd4isc26::detail
