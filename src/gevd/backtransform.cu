#include "gevd/backtransform.hpp"

#include "runtime/error.hpp"

#include <algorithm>
#include <stdexcept>

namespace gevd4isc26::detail {
namespace {

__device__ std::int64_t localToGlobal(std::int64_t local_index,
                                      std::int64_t block_size,
                                      int process_coordinate,
                                      int source_coordinate,
                                      int process_count) {
  const std::int64_t local_block = local_index / block_size;
  const std::int64_t offset_in_block = local_index % block_size;
  const int first_owned_block =
      (process_coordinate - source_coordinate + process_count) % process_count;
  return (local_block * process_count + first_owned_block) * block_size +
         offset_in_block;
}

__global__ void fillDistributedIdentity(double* matrix,
                                        std::int64_t leading_dimension,
                                        std::int64_t local_columns,
                                        std::int64_t order,
                                        std::int64_t row_block_size,
                                        std::int64_t column_block_size,
                                        int source_row,
                                        int source_column,
                                        int grid_rows,
                                        int grid_columns,
                                        int process_row,
                                        int process_column) {
  const std::int64_t elements = leading_dimension * local_columns;
  const std::int64_t stride =
      static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += stride) {
    const std::int64_t local_row = index % leading_dimension;
    const std::int64_t local_column = index / leading_dimension;
    const std::int64_t global_row =
        localToGlobal(local_row, row_block_size, process_row, source_row,
                      grid_rows);
    const std::int64_t global_column =
        localToGlobal(local_column, column_block_size, process_column,
                      source_column, grid_columns);
    matrix[index] = global_row < order && global_column < order &&
                            global_row == global_column
                        ? 1.0
                        : 0.0;
  }
}

void queryTriangularSolve(const DistributedGpuContext& context,
                          std::int64_t order,
                          double* cholesky_factor,
                          double* right_hand_side,
                          cublasMpMatrixDescriptor_t cholesky_descriptor,
                          cublasMpMatrixDescriptor_t rhs_descriptor,
                          WorkspaceSize* workspace) {
  const double alpha = 1.0;
  std::size_t device_bytes = 0;
  std::size_t host_bytes = 0;
  GEVD_CUBLASMP(cublasMpTrsm_bufferSize(
      context.blas(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
      CUBLAS_DIAG_NON_UNIT, order, order, &alpha, cholesky_factor, 1, 1,
      cholesky_descriptor, right_hand_side, 1, 1, rhs_descriptor,
      CUBLAS_COMPUTE_64F, &device_bytes, &host_bytes));
  workspace->include(device_bytes, host_bytes);
}

}  // namespace

WorkspaceSize queryDirectBacktransformWorkspace(
    const DistributedGpuContext& context,
    std::int64_t order,
    double* cholesky_factor,
    double* eigenvectors,
    cublasMpMatrixDescriptor_t cholesky_descriptor,
    cublasMpMatrixDescriptor_t eigenvector_descriptor) {
  WorkspaceSize workspace;
  queryTriangularSolve(context, order, cholesky_factor, eigenvectors,
                       cholesky_descriptor, eigenvector_descriptor, &workspace);
  return workspace;
}

WorkspaceSize queryOverlapBacktransformWorkspace(
    const DistributedGpuContext& context,
    std::int64_t order,
    double* cholesky_factor,
    double* inverse_transpose,
    double* eigenvectors,
    double* output,
    cublasMpMatrixDescriptor_t cholesky_descriptor,
    cublasMpMatrixDescriptor_t matrix_descriptor,
    bool inverse_is_cached) {
  WorkspaceSize workspace;
  if (!inverse_is_cached) {
    queryTriangularSolve(context, order, cholesky_factor, inverse_transpose,
                         cholesky_descriptor, matrix_descriptor, &workspace);
  }

  const double alpha = 1.0;
  const double beta = 0.0;
  std::size_t device_bytes = 0;
  std::size_t host_bytes = 0;
  GEVD_CUBLASMP(cublasMpGemm_bufferSize(
      context.blas(), CUBLAS_OP_N, CUBLAS_OP_N, order, order, order, &alpha,
      inverse_transpose, 1, 1, matrix_descriptor, eigenvectors, 1, 1,
      matrix_descriptor, &beta, output, 1, 1, matrix_descriptor,
      CUBLAS_COMPUTE_64F, &device_bytes, &host_bytes));
  workspace.include(device_bytes, host_bytes);
  return workspace;
}

void backtransformWithTriangularSolve(
    const DistributedGpuContext& context,
    std::int64_t order,
    double* cholesky_factor,
    double* eigenvectors,
    cublasMpMatrixDescriptor_t cholesky_descriptor,
    cublasMpMatrixDescriptor_t eigenvector_descriptor,
    DeviceAllocation& device_workspace,
    HostAllocation& host_workspace) {
  const double alpha = 1.0;
  GEVD_CUBLASMP(cublasMpTrsm(
      context.blas(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
      CUBLAS_DIAG_NON_UNIT, order, order, &alpha, cholesky_factor, 1, 1,
      cholesky_descriptor, eigenvectors, 1, 1, eigenvector_descriptor,
      CUBLAS_COMPUTE_64F, device_workspace.data(), device_workspace.bytes(),
      host_workspace.data(), host_workspace.bytes()));
}

OverlapInverseBacktransform::OverlapInverseBacktransform(
    const DistributedGpuContext& context,
    const MatrixDistribution& distribution,
    std::int64_t leading_dimension,
    double* cholesky_factor,
    double* inverse_transpose,
    cublasMpMatrixDescriptor_t cholesky_descriptor,
    cublasMpMatrixDescriptor_t matrix_descriptor,
    DeviceAllocation& device_workspace,
    HostAllocation& host_workspace,
    bool inverse_is_cached)
    : context_(context),
      distribution_(distribution),
      leading_dimension_(leading_dimension),
      cholesky_factor_(cholesky_factor),
      inverse_transpose_(inverse_transpose),
      cholesky_descriptor_(cholesky_descriptor),
      matrix_descriptor_(matrix_descriptor),
      device_workspace_(device_workspace),
      host_workspace_(host_workspace),
      inverse_is_cached_(inverse_is_cached) {}

OverlapInverseBacktransform::~OverlapInverseBacktransform() {
  if (inverse_start_ != nullptr) {
    cudaEventDestroy(inverse_start_);
  }
  if (inverse_end_ != nullptr) {
    cudaEventDestroy(inverse_end_);
  }
}

int (*OverlapInverseBacktransform::callback() const noexcept)(void*) {
  return &OverlapInverseBacktransform::invoke;
}

int OverlapInverseBacktransform::invoke(void* user_data) noexcept {
  if (user_data == nullptr) {
    return 1;
  }
  return static_cast<OverlapInverseBacktransform*>(user_data)->launch();
}

int OverlapInverseBacktransform::launch() noexcept {
  launched_ = true;
  if (inverse_is_cached_) {
    return 0;
  }

  cuda_status_ = cudaSetDevice(context_.device());
  if (cuda_status_ != cudaSuccess) {
    return 1;
  }

  cuda_status_ = cudaEventCreate(&inverse_start_);
  if (cuda_status_ != cudaSuccess) {
    return 1;
  }
  cuda_status_ = cudaEventCreate(&inverse_end_);
  if (cuda_status_ != cudaSuccess) {
    return 1;
  }
  cuda_status_ = cudaEventRecord(inverse_start_, context_.stream());
  if (cuda_status_ != cudaSuccess) {
    return 1;
  }

  constexpr int threads = 256;
  constexpr int blocks = 4096;
  const ProcessGrid& grid = distribution_.grid();
  fillDistributedIdentity<<<blocks, threads, 0, context_.stream()>>>(
      inverse_transpose_, leading_dimension_, distribution_.localColumns(),
      distribution_.order(), distribution_.rowBlockSize(),
      distribution_.columnBlockSize(), grid.source_row, grid.source_column,
      grid.rows, grid.columns, distribution_.processRow(),
      distribution_.processColumn());
  cuda_status_ = cudaGetLastError();
  if (cuda_status_ != cudaSuccess) {
    return 1;
  }

  const double alpha = 1.0;
  cublas_status_ = cublasMpTrsm(
      context_.blas(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
      CUBLAS_DIAG_NON_UNIT, distribution_.order(), distribution_.order(),
      &alpha, cholesky_factor_, 1, 1, cholesky_descriptor_,
      inverse_transpose_, 1, 1, matrix_descriptor_, CUBLAS_COMPUTE_64F,
      device_workspace_.data(), device_workspace_.bytes(),
      host_workspace_.data(), host_workspace_.bytes());
  if (cublas_status_ != CUBLASMP_STATUS_SUCCESS) {
    return 1;
  }
  cuda_status_ = cudaEventRecord(inverse_end_, context_.stream());
  return cuda_status_ == cudaSuccess ? 0 : 1;
}

void OverlapInverseBacktransform::wait() {
  if (!launched_) {
    throw std::logic_error(
        "the standard EVD stage did not invoke concurrent inverse preparation");
  }
  GEVD_CUDA(cudaSetDevice(context_.device()));
  GEVD_CUDA(cudaStreamSynchronize(context_.stream()));
  GEVD_CUDA(cuda_status_);
  GEVD_CUBLASMP(cublas_status_);
  if (!inverse_is_cached_) {
    GEVD_CUDA(cudaEventElapsedTime(&inverse_milliseconds_, inverse_start_,
                                   inverse_end_));
  }
}

void OverlapInverseBacktransform::multiply(double* eigenvectors,
                                           double* output) {
  if (!launched_) {
    throw std::logic_error("inverse backtransform was not launched");
  }
  const double alpha = 1.0;
  const double beta = 0.0;
  const std::int64_t order = distribution_.order();
  GEVD_CUBLASMP(cublasMpGemm(
      context_.blas(), CUBLAS_OP_N, CUBLAS_OP_N, order, order, order, &alpha,
      inverse_transpose_, 1, 1, matrix_descriptor_, eigenvectors, 1, 1,
      matrix_descriptor_, &beta, output, 1, 1, matrix_descriptor_,
      CUBLAS_COMPUTE_64F, device_workspace_.data(), device_workspace_.bytes(),
      host_workspace_.data(), host_workspace_.bytes()));
}

bool OverlapInverseBacktransform::inverseWasComputed() const noexcept {
  return launched_ && !inverse_is_cached_;
}

double OverlapInverseBacktransform::inversePreparationSeconds() const noexcept {
  return static_cast<double>(inverse_milliseconds_) / 1000.0;
}

}  // namespace gevd4isc26::detail
