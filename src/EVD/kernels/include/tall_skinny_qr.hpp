#pragma once

#include <cusolverDn.h>

#include <iostream>
#include <type_traits>

#include "qr_kernels.cuh"
#include "runtime/cuda_check.hpp"

template <typename T, long M, long N>
void factorTallSkinnyPanelQr(cublasHandle_t cublas_handle,
                             long m,
                             long n,
                             T* A,
                             long lda,
                             T* R,
                             long ldr,
                             T* work)
{
  if (n > N)
  {
    std::cerr << "Tall-skinny QR panel width exceeds the compiled limit"
              << std::endl;
    ::gevd4isc26::evd::abortActiveCommunicator(1);
  }


  dim3 blockDim(32, 16);

  if (m <= M)
  {

    householderQrKernel<M, N><<<1, blockDim>>>(m, n, A, lda, R, ldr);
    EVD_CUDA_CHECK(cudaGetLastError());
    EVD_CUDA_CHECK(cudaDeviceSynchronize());
    return;
  }

  if (0 != (m % M) % n)
  {
    std::cerr << "Tall-skinny QR remainder must be divisible by panel width"
              << std::endl;
    ::gevd4isc26::evd::abortActiveCommunicator(1);
  }

  cudaDataType_t cuda_data_type;
  cublasComputeType_t cublas_compute_type;

  if (std::is_same<T, double>::value)
  {
    cuda_data_type      = CUDA_R_64F;
    cublas_compute_type = CUBLAS_COMPUTE_64F;
  }
  else if (std::is_same<T, float>::value)
  {
    cuda_data_type      = CUDA_R_32F;
    cublas_compute_type = CUBLAS_COMPUTE_32F;
  }
  else if (std::is_same<T, half>::value)
  {
    cuda_data_type      = CUDA_R_16F;
    cublas_compute_type = CUBLAS_COMPUTE_16F;
  }


  long blockNum = (m + M - 1) / M;
  long ldwork   = blockNum * n;

  householderQrKernel<M, N><<<blockNum, blockDim>>>(m, n, A, lda, work, ldwork);


  factorTallSkinnyPanelQr<T, M, N>(cublas_handle, ldwork, n, work, ldwork, R, ldr, work + n * ldwork);

  T tone = 1.0, tzero = 0.0;
  cublasGemmStridedBatchedEx(cublas_handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             M,
                             n,
                             n,
                             &tone,
                             A,
                             cuda_data_type,
                             lda,
                             M,

                             work,
                             cuda_data_type,
                             ldwork,
                             n,

                             &tzero,

                             A,
                             cuda_data_type,
                             lda,
                             M,
                             m / M,

                             cublas_compute_type,
                             CUBLAS_GEMM_DEFAULT);


  long mm = m % M;
  if (0 < mm)
  {
    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_N,
                 CUBLAS_OP_N,
                 mm,
                 n,
                 n,
                 &tone,
                 A + (m - mm),
                 cuda_data_type,
                 lda,

                 work + (m / M * n),
                 cuda_data_type,
                 ldwork,

                 &tzero,
                 A + (m - mm),
                 cuda_data_type,
                 lda,

                 cublas_compute_type,
                 CUBLAS_GEMM_DEFAULT);


  }

}
