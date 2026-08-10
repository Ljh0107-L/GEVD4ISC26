// Source-derived optimized symmetric rank-2k implementation.
// #inclue "cuSolver.h"
#include "matrix_partition.hpp"
#include <cusolverDn.h>



template <typename T>
static __global__ void setInitialValue(long int m, long int n, T *a, long int lda, T val)
{
  long int i = threadIdx.x + blockDim.x * blockIdx.x;
  long int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < n)
  {
    a[i + j * lda] = val;
  }
}

template <typename T>
static __global__ void matrixCpy(long int m, long int n, T *a, long int lda, T *b, long int ldb)
{
  long int i = threadIdx.x + blockDim.x * blockIdx.x;
  long int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < n)
  {
    b[i + j * ldb] = a[i + j * lda];
  }
}

__global__ static void
matrixCpyH2F(long int m, long int n, half *a, long int lda, float *b, long int ldb)
{
  long int i = threadIdx.x + blockDim.x * blockIdx.x;
  long int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < n)
  {
    b[i + j * ldb] = __half2float(a[i + j * lda]);
  }
}

__global__ static void
matrixCpyF2H(long int m, long int n, float *a, long int lda, half *b, long int ldb)
{
  long int i = threadIdx.x + blockDim.x * blockIdx.x;
  long int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < n)
  {
    b[i + j * ldb] = __float2half(a[i + j * lda]);
  }
}


template <typename T>
void ozimmuSymmetricRank2kPhase2(cublasHandle_t handle,
                        long int n,
                        long int k,
                        T alpha,
                        T *A,
                        long int lda,
                        T *B,
                        long int ldb,
                        T beta,
                        T *C,
                        long int ldc,
                        long int nb)
{
  T tOne = 1.0;

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


  cublasGemmStridedBatchedEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_T,
                             nb,
                             nb,
                             k,
                             &alpha,
                             A,
                             cuda_data_type,
                             lda,
                             nb,
                             B,
                             cuda_data_type,
                             ldb,
                             nb,
                             &beta,
                             C,
                             cuda_data_type,
                             ldc,
                             nb + nb * ldc,
                             n / nb,
                             cublas_compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  

  cublasGemmStridedBatchedEx(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_T,
                             nb,
                             nb,
                             k,
                             &alpha,
                             B,
                             cuda_data_type,
                             ldb,
                             nb,
                             A,
                             cuda_data_type,
                             lda,
                             nb,
                             &tOne,
                             C,
                             cuda_data_type,
                             ldc,
                             nb + nb * ldc,

                             n / nb,
                             cublas_compute_type,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  for (int i = 1; n / nb / i / 2 >= 1; i *= 2)
  {

    cublasGemmStridedBatchedEx(handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_T,
                               i * nb,
                               i * nb,
                               k,
                               &alpha,
                               A + i * nb,
                               cuda_data_type,
                               lda,
                               2 * i * nb,
                               B,
                               cuda_data_type,
                               ldb,
                               2 * i * nb,
                               &beta,
                               C + i * nb,
                               cuda_data_type,
                               ldc,
                               2 * (i * nb + i * nb * ldc),
                               n / nb / i / 2,
                               cublas_compute_type,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cublasGemmStridedBatchedEx(handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_T,
                               i * nb,
                               i * nb,
                               k,
                               &alpha,
                               B + i * nb,
                               cuda_data_type,
                               ldb,
                               2 * i * nb,
                               A,
                               cuda_data_type,
                               lda,
                               2 * i * nb,
                               &tOne,
                               C + i * nb,
                               cuda_data_type,
                               ldc,
                               2 * (i * nb + i * nb * ldc),
                               n / nb / i / 2,
                               cublas_compute_type,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
}


template <typename T>
void ozimmuSymmetricRank2kPhase3(cublasHandle_t handle,
                        long int n,
                        long int k,
                        T alpha,
                        T *A,
                        long int lda,
                        T *B,
                        long int ldb,
                        T beta,
                        T *C,
                        long int ldc,
                        long int nb);

template <>
void ozimmuSymmetricRank2kPhase3(cublasHandle_t handle,
                        long int n,
                        long int k,
                        double alpha,
                        double *A,
                        long int lda,
                        double *B,
                        long int ldb,
                        double beta,
                        double *C,
                        long int ldc,
                        long int nb)
{

  double done = 1.0;

  const std::vector<std::int64_t> matrixPartitions =
      partitionSymmetricRank2k(n);
  const int length = static_cast<int>(matrixPartitions.size()) - 1;
  const std::int64_t* matSize = matrixPartitions.data();
  int offset;
  int rest_n = n;

  cudaDataType_t cuda_data_type;
  cublasComputeType_t cublas_compute_type;

  cuda_data_type      = CUDA_R_64F;
  cublas_compute_type = CUBLAS_COMPUTE_64F;

  // printf("n=%ld, k=%ld,nb=%ld,length=%d.\n", n, k, nb, length);

  for (int i = length; i >= 0; i--)
  {

    int nn = matSize[i];

    if (i < length)
      offset += matSize[i + 1];
    else
      offset = 0;

    // printf("i=%d, nn = %d, offset=%d, alpha = %lf, beta = %lf.\n", i, nn, offset, alpha, beta);

    if (nn % 8192 == 0)
    {

      ozimmuSymmetricRank2kPhase2(handle,
                         nn,
                         k,
                         alpha,
                         A + offset,
                         lda,
                         B + offset,
                         ldb,
                         beta,
                         C + offset + offset * ldc,
                         ldc,
                         nb);
     
    }
    else
    {

      cublasDsyr2k(handle,
                   CUBLAS_FILL_MODE_LOWER,
                   CUBLAS_OP_N,
                   nn,
                   k,
                   &alpha,
                   A + offset,
                   lda,
                   B + offset,
                   ldb,
                   &beta,
                   C + offset + offset * ldc,
                   ldc);

     
    }
    if (i != 0)
    {
      rest_n -= nn;

      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   A + offset + nn,
                   cuda_data_type,
                   lda,
                   B + offset,
                   cuda_data_type,
                   ldb,

                   &beta,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,

                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);



      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   B + offset + nn,
                   cuda_data_type,
                   ldb,
                   A + offset,
                   cuda_data_type,
                   lda,
                   &done,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);


    }
    else
      return;
  }
  return;
}

template <>
void ozimmuSymmetricRank2kPhase3(cublasHandle_t handle,
                        long int n,
                        long int k,
                        float alpha,
                        float *A,
                        long int lda,
                        float *B,
                        long int ldb,
                        float beta,
                        float *C,
                        long int ldc,
                        long int nb)
{

  const std::vector<std::int64_t> matrixPartitions =
      partitionSymmetricRank2k(n);
  const int length = static_cast<int>(matrixPartitions.size()) - 1;
  const std::int64_t* matSize = matrixPartitions.data();
  int offset;
  int rest_n = n;

  float fone = 1.0;

  cudaDataType_t cuda_data_type;
  cublasComputeType_t cublas_compute_type;

  cuda_data_type      = CUDA_R_32F;
  cublas_compute_type = CUBLAS_COMPUTE_32F;

  // printf("n=%ld, k=%ld,nb=%ld,length=%d.\n", n, k, nb, length);

  for (int i = length; i >= 0; i--)
  {

    int nn = matSize[i];

    if (i < length)
      offset += matSize[i + 1];
    else
      offset = 0;

    // printf("i=%d, nn = %d, offset=%d, alpha = %lf, beta = %lf.\n", i, nn, offset, alpha, beta);

    if (nn % 8192 == 0)
    {


      ozimmuSymmetricRank2kPhase2(handle,
                         nn,
                         k,
                         alpha,
                         A + offset,
                         lda,
                         B + offset,
                         ldb,
                         beta,
                         C + offset + offset * ldc,
                         ldc,
                         nb);

    }
    else
    {

      cublasSsyr2k(handle,
                   CUBLAS_FILL_MODE_LOWER,
                   CUBLAS_OP_N,
                   nn,
                   k,
                   &alpha,
                   A + offset,
                   lda,
                   B + offset,
                   ldb,
                   &beta,
                   C + offset + offset * ldc,
                   ldc);

    }
    if (i != 0)
    {
      rest_n -= nn;

     
      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   A + offset + nn,
                   cuda_data_type,
                   lda,
                   B + offset,
                   cuda_data_type,
                   ldb,

                   &beta,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,

                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);



      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   B + offset + nn,
                   cuda_data_type,
                   ldb,
                   A + offset,
                   cuda_data_type,
                   lda,
                   &fone,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    }
    else
      return;
  }
  return;
}

template <>
void ozimmuSymmetricRank2kPhase3(cublasHandle_t handle,
                        long int n,
                        long int k,
                        half alpha,
                        half *A,
                        long int lda,
                        half *B,
                        long int ldb,
                        half beta,
                        half *C,
                        long int ldc,
                        long int nb)
{

  const std::vector<std::int64_t> matrixPartitions =
      partitionSymmetricRank2k(n);
  const int length = static_cast<int>(matrixPartitions.size()) - 1;
  const std::int64_t* matSize = matrixPartitions.data();
  int offset;
  int rest_n = n;

  half hone = 1.0;

  cudaDataType_t cuda_data_type;
  cublasComputeType_t cublas_compute_type;

  cuda_data_type      = CUDA_R_16F;
  cublas_compute_type = CUBLAS_COMPUTE_16F;

  // printf("n=%ld, k=%ld,nb=%ld,length=%d.\n", n, k, nb, length);

  for (int i = length; i >= 0; i--)
  {

    int nn = matSize[i];

    if (i < length)
      offset += matSize[i + 1];
    else
      offset = 0;

    // printf("i=%d, nn = %d, offset=%d, alpha = %lf, beta = %lf.\n", i, nn, offset, alpha, beta);

    if (nn % 8192 == 0)
    {

      ozimmuSymmetricRank2kPhase2(handle,
                         nn,
                         k,
                         alpha,
                         A + offset,
                         lda,
                         B + offset,
                         ldb,
                         beta,
                         C + offset + offset * ldc,
                         ldc,
                         nb);

    }
    else
    {

      float *A_, *B_, *C_;

      float sAlpha = __half2float(alpha);
      float sBeta  = __half2float(beta);

      cudaMalloc(&A_, sizeof(float) * nn * k);
      cudaMalloc(&B_, sizeof(float) * nn * k);
      cudaMalloc(&C_, sizeof(float) * nn * nn);
      //   printf("%ld, %ld\n", n, k);
      dim3 grid1((nn + 31) / 32, (k + 31) / 32);
      dim3 block(32, 32);

      matrixCpyH2F<<<grid1, block>>>(nn, k, A + offset, lda, A_, nn);
      matrixCpyH2F<<<grid1, block>>>(nn, k, B + offset, ldb, B_, nn);
      matrixCpyH2F<<<grid1, block>>>(nn, nn, C + offset + offset * ldc, ldc, C_, nn);

      cublasSsyr2k(handle,
                   CUBLAS_FILL_MODE_LOWER,
                   CUBLAS_OP_N,
                   nn,
                   k,
                   &sAlpha,
                   A_,
                   nn,
                   B_,
                   nn,
                   &sBeta,
                   C_,
                   nn);

      matrixCpyF2H<<<grid1, block>>>(nn, k, C_, nn, C + offset + offset * ldc, ldc);

      cudaFree(A_);
      cudaFree(B_);
      cudaFree(C_);

    }

    if (i != 0)
    {
      rest_n -= nn;

      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   A + offset + nn,
                   cuda_data_type,
                   lda,
                   B + offset,
                   cuda_data_type,
                   ldb,

                   &beta,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,

                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);


      cublasGemmEx(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   rest_n,
                   nn,
                   k,
                   &alpha,
                   B + offset + nn,
                   cuda_data_type,
                   ldb,
                   A + offset,
                   cuda_data_type,
                   lda,
                   &hone,
                   C + offset + offset * ldc + nn,
                   cuda_data_type,
                   ldc,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    }
    else
      return;
  }
  return;
}

template <typename T>
void tensorCoreSymmetricRank2k(cublasHandle_t handle,
                     long int n,
                     long int k,
                     T alpha,
                     T *A,
                     long int lda,
                     T *B,
                     long int ldb,
                     T beta,
                     T *C,
                     long int ldc,
                     long int nb)
{
  if (n % 2 || k % 2)
  {
    T *A_, *C_, *B_;
    long int N = n, K = k, lda_, ldb_, ldc_;
    n += n % 2;
    k += k % 2;
    lda_ = lda + lda % 2;
    ldb_ = ldb + ldb % 2;
    ldc_ = ldc + ldc % 2;
    cudaMalloc(&A_, sizeof(T) * n * k);
    cudaMalloc(&B_, sizeof(T) * n * k);
    cudaMalloc(&C_, sizeof(T) * n * n);
    printf("%ld, %ld\n", n, k);
    dim3 grid1((n + 31) / 32, (k + 31) / 32);
    dim3 block(32, 32);
    setInitialValue<<<grid1, block>>>(n, k, A_, lda_, T(0.0));
    setInitialValue<<<grid1, block>>>(n, k, B_, ldb_, T(0.0));
    dim3 grid2((n + 31) / 32, (n + 31) / 32);
    setInitialValue<<<grid2, block>>>(n, n, C_, ldc_, T(1.0));
    dim3 grid3((N + 31) / 32, (K + 31) / 32);
    matrixCpy<<<grid3, block>>>(N, K, A, lda, A_, lda_);
    matrixCpy<<<grid3, block>>>(N, K, B, ldb, B_, ldb_);

    ozimmuSymmetricRank2kPhase3(handle, n, k, alpha, A_, lda_, B_, ldb_, beta, C_, ldc_, nb);
    dim3 grid4((N + 31) / 32, (N + 31) / 32);
    matrixCpy<<<grid4, block>>>(N, N, C_, ldc_, C, ldc);

    printf("check ok\n");
    cudaFree(A_);
    cudaFree(B_);
    cudaFree(C_);
  }
  else
  {
    ozimmuSymmetricRank2kPhase3(handle, n, k, alpha, A, lda, B, ldb, beta, C, ldc, nb);
  }
}

template void tensorCoreSymmetricRank2k(cublasHandle_t handle,
                              long int n,
                              long int k,
                              double alpha,
                              double *A,
                              long int lda,
                              double *B,
                              long int ldb,
                              double beta,
                              double *C,
                              long int ldc,
                              long int nb);
template void tensorCoreSymmetricRank2k(cublasHandle_t handle,
                              long int n,
                              long int k,
                              float alpha,
                              float *A,
                              long int lda,
                              float *B,
                              long int ldb,
                              float beta,
                              float *C,
                              long int ldc,
                              long int nb);

template void tensorCoreSymmetricRank2k(cublasHandle_t handle,
                              long int n,
                              long int k,
                              half alpha,
                              half *A,
                              long int lda,
                              half *B,
                              long int ldb,
                              half beta,
                              half *C,
                              long int ldc,
                              long int nb);
