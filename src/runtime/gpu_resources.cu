#include "runtime/gpu_resources.hpp"

#include "runtime/cpu_affinity.hpp"
#include "runtime/error.hpp"

namespace gevd4isc26::detail {

CudaStream::~CudaStream() {
  reset();
}

void CudaStream::create(int device) {
  reset();
  device_ = device;
  GEVD_CUDA(cudaSetDevice(device_));
  GEVD_CUDA(cudaStreamCreate(&stream_));
}

void CudaStream::reset() noexcept {
  if (stream_ != nullptr) {
    if (device_ >= 0) {
      cudaSetDevice(device_);
    }
    cudaStreamDestroy(stream_);
  }
  stream_ = nullptr;
  device_ = -1;
}

NcclCommunicator::~NcclCommunicator() {
  reset();
}

void NcclCommunicator::create(MPI_Comm communicator, int rank, int size) {
  [[maybe_unused]] ScopedCpuAffinityRestore preserve_cpu_affinity;
  reset();
  ncclUniqueId id;
  if (rank == 0) {
    GEVD_NCCL(ncclGetUniqueId(&id));
  }
  GEVD_MPI(MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0,
                     communicator));
  GEVD_NCCL(ncclCommInitRank(&communicator_, size, id, rank));
}

void NcclCommunicator::reset() noexcept {
  if (communicator_ != nullptr) {
    ncclCommDestroy(communicator_);
  }
  communicator_ = nullptr;
}

SolverMpResources::~SolverMpResources() {
  reset();
}

void SolverMpResources::create(int device,
                               cudaStream_t stream,
                               ncclComm_t nccl,
                               int grid_rows,
                               int grid_columns) {
  [[maybe_unused]] ScopedCpuAffinityRestore preserve_cpu_affinity;
  reset();
  GEVD_CUSOLVER(cusolverMpCreate(&handle_, device, stream));
  try {
    GEVD_CUSOLVER(cusolverMpCreateDeviceGrid(
        handle_, &grid_, nccl, grid_rows, grid_columns,
        CUSOLVERMP_GRID_MAPPING_ROW_MAJOR));
  } catch (...) {
    reset();
    throw;
  }
}

void SolverMpResources::reset() noexcept {
  if (grid_ != nullptr) {
    cusolverMpDestroyGrid(grid_);
  }
  grid_ = nullptr;
  if (handle_ != nullptr) {
    cusolverMpDestroy(handle_);
  }
  handle_ = nullptr;
}

BlasMpResources::~BlasMpResources() {
  reset();
}

void BlasMpResources::create(cudaStream_t stream,
                             ncclComm_t nccl,
                             int grid_rows,
                             int grid_columns) {
  [[maybe_unused]] ScopedCpuAffinityRestore preserve_cpu_affinity;
  reset();
  GEVD_CUBLASMP(cublasMpCreate(&handle_, stream));
  try {
    GEVD_CUBLASMP(cublasMpGridCreate(
        grid_rows, grid_columns, CUBLASMP_GRID_LAYOUT_ROW_MAJOR, nccl,
        &grid_));
  } catch (...) {
    reset();
    throw;
  }
}

void BlasMpResources::reset() noexcept {
  if (grid_ != nullptr) {
    cublasMpGridDestroy(grid_);
  }
  grid_ = nullptr;
  if (handle_ != nullptr) {
    cublasMpDestroy(handle_);
  }
  handle_ = nullptr;
}

}  // namespace gevd4isc26::detail
