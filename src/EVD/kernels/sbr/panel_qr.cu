#include <cstdlib>
#include <string>
#include <vector>

#include <curand.h>
#include <cusolverDn.h>

#include "qr_kernels.cuh"
#include "tall_skinny_qr.hpp"
#include "utility_kernels.cuh"

namespace {

float panelQrMilliseconds = 0.0F;
float panelUpdateMilliseconds = 0.0F;

bool profilePanelQrEnabled()
{
  static const bool enabled = []() {
    const char *value = std::getenv("EVD_PROFILE_PANEL_QR");
    return value != nullptr && std::atoi(value) != 0;
  }();
  return enabled;
}

void startPanelQrTimerIfEnabled()
{
  if (profilePanelQrEnabled()) {
    startTimer();
  }
}

void accumulatePanelQrTimeIfEnabled(float *totalMilliseconds)
{
  if (profilePanelQrEnabled()) {
    *totalMilliseconds += stopTimer();
  }
}

} // namespace

template <typename T>
void factorPanelQr(cusolverDnHandle_t cusolver_handle,
             cublasHandle_t cublas_handle,
             long m,
             long n,
             T *A,
             long lda,
             T *W,
             long ldw,
             T *R,
             long ldr,
             T *work,
             int *info)
{

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

  if (n <= 32)
  {
    startPanelQrTimerIfEnabled();

    factorTallSkinnyPanelQr<T, 128, 32>(cublas_handle, m, n, A, lda, R, ldr, work);


    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);

    launchIdentityMinusMatrix(gridDim, blockDim, m, n, A, lda);


    launchCopyMatrix(gridDim, blockDim, m, n, A, lda, W, ldw);


    cusolverDnDgetrf(cusolver_handle, m, n, A, lda, work, NULL, info);


    launchKeepLowerTriangle(gridDim, blockDim, m, n, A, lda);
    // launchClearMatrix(gridDim, blockDim, m, n, W, lda);


    double done = 1.0;
    cublasDtrsm(cublas_handle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                m,
                n,
                &done,
                A,
                lda,
                W,
                ldw);


    accumulatePanelQrTimeIfEnabled(&panelQrMilliseconds);

    return;
  }

  factorPanelQr(cusolver_handle, cublas_handle, m, n / 2, A, lda, W, ldw, R, ldr, work, info);


  T tone    = 1.0;
  T tzero   = 0.0;
  T tnegone = -1.0;

  startPanelQrTimerIfEnabled();


  cublasGemmEx(cublas_handle,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               n / 2,
               n - n / 2,
               m,
               &tone,
               W,
               cuda_data_type,
               ldw,

               A + n / 2 * lda,
               cuda_data_type,
               lda,

               &tzero,
               work,
               cuda_data_type,
               n / 2,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  cublasGemmEx(cublas_handle,
               CUBLAS_OP_N,
               CUBLAS_OP_N,
               m,
               n - n / 2,
               n / 2,
               &tnegone,
               A,
               cuda_data_type,
               lda,

               work,
               cuda_data_type,
               n / 2,

               &tone,
               A + n / 2 * lda,
               cuda_data_type,
               lda,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  accumulatePanelQrTimeIfEnabled(&panelUpdateMilliseconds);



  dim3 gridDim((n / 2 + 32 - 1) / 32, (n - n / 2 + 32 - 1) / 32);
  dim3 blockDim(32, 32);

  launchCopyAndClear(gridDim,
                            blockDim,
                            n / 2,
                            n - n / 2,
                            A + n / 2 * lda,
                            lda,
                            R + n / 2 * ldr,
                            ldr);


  factorPanelQr(cusolver_handle,
          cublas_handle,
          m - n / 2,
          n - n / 2,
          A + n / 2 + n / 2 * lda,
          lda,
          W + n / 2 + n / 2 * ldw,
          ldw,
          R + n / 2 + n / 2 * ldr,
          ldr,
          work,
          info);


  startPanelQrTimerIfEnabled();
  cublasGemmEx(cublas_handle,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               n / 2,
               n - n / 2,
               m - n / 2,
               &tone,
               A + n / 2,
               cuda_data_type,
               lda,

               W + n / 2 + n / 2 * ldw,
               cuda_data_type,
               ldw,

               &tzero,
               work,
               cuda_data_type,
               n / 2,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  cublasGemmEx(cublas_handle,
               CUBLAS_OP_N,
               CUBLAS_OP_N,
               m,
               n - n / 2,
               n / 2,
               &tnegone,
               W,
               cuda_data_type,
               ldw,

               work,
               cuda_data_type,
               n / 2,

               &tone,
               W + n / 2 * ldw,
               cuda_data_type,
               ldw,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);

  accumulatePanelQrTimeIfEnabled(&panelUpdateMilliseconds);

  return;
}


template void factorPanelQr<double>(cusolverDnHandle_t,
                              cublasHandle_t,
                              long,
                              long,
                              double *,
                              long,
                              double *,
                              long,
                              double *,
                              long,
                              double *,
                              int *);

template <>
void factorPanelQr(cusolverDnHandle_t cusolver_handle,
             cublasHandle_t cublas_handle,
             long m,
             long n,
             float *A,
             long lda,
             float *W,
             long ldw,
             float *R,
             long ldr,
             float *work,
             int *info)
{

  cudaDataType_t cuda_data_type;
  cublasComputeType_t cublas_compute_type;

  cuda_data_type      = CUDA_R_32F;
  cublas_compute_type = CUBLAS_COMPUTE_32F;

  if (n <= 32)
  {
    startPanelQrTimerIfEnabled();

    factorTallSkinnyPanelQr<float, 128, 32>(cublas_handle, m, n, A, lda, R, ldr, work);


    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);

    launchIdentityMinusMatrix(gridDim, blockDim, m, n, A, lda);


    launchCopyMatrix(gridDim, blockDim, m, n, A, lda, W, ldw);

    cusolverDnSgetrf(cusolver_handle, m, n, A, lda, work, NULL, info);


    launchKeepLowerTriangle(gridDim, blockDim, m, n, A, lda);
    // launchClearMatrix(gridDim, blockDim, m, n, W, lda);


    float fone = 1.0;
    cublasStrsm(cublas_handle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                m,
                n,
                &fone,
                A,
                lda,
                W,
                ldw);


    accumulatePanelQrTimeIfEnabled(&panelQrMilliseconds);

    return;
  }

  factorPanelQr(cusolver_handle, cublas_handle, m, n / 2, A, lda, W, ldw, R, ldr, work, info);


  float tone    = 1.0;
  float tzero   = 0.0;
  float tnegone = -1.0;

  startPanelQrTimerIfEnabled();

  cublasGemmEx(cublas_handle,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               n / 2,
               n - n / 2,
               m,
               &tone,
               W,
               cuda_data_type,
               ldw,

               A + n / 2 * lda,
               cuda_data_type,
               lda,

               &tzero,
               work,
               cuda_data_type,
               n / 2,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  cublasGemmEx(cublas_handle,
               CUBLAS_OP_N,
               CUBLAS_OP_N,
               m,
               n - n / 2,
               n / 2,
               &tnegone,
               A,
               cuda_data_type,
               lda,

               work,
               cuda_data_type,
               n / 2,

               &tone,
               A + n / 2 * lda,
               cuda_data_type,
               lda,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  accumulatePanelQrTimeIfEnabled(&panelUpdateMilliseconds);



  dim3 gridDim((n / 2 + 32 - 1) / 32, (n - n / 2 + 32 - 1) / 32);
  dim3 blockDim(32, 32);

  launchCopyAndClear(gridDim,
                            blockDim,
                            n / 2,
                            n - n / 2,
                            A + n / 2 * lda,
                            lda,
                            R + n / 2 * ldr,
                            ldr);

  factorPanelQr(cusolver_handle,
          cublas_handle,
          m - n / 2,
          n - n / 2,
          A + n / 2 + n / 2 * lda,
          lda,
          W + n / 2 + n / 2 * ldw,
          ldw,
          R + n / 2 + n / 2 * ldr,
          ldr,
          work,
          info);


  startPanelQrTimerIfEnabled();
  cublasGemmEx(cublas_handle,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               n / 2,
               n - n / 2,
               m - n / 2,
               &tone,
               A + n / 2,
               cuda_data_type,
               lda,

               W + n / 2 + n / 2 * ldw,
               cuda_data_type,
               ldw,

               &tzero,
               work,
               cuda_data_type,
               n / 2,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);

  cublasGemmEx(cublas_handle,
               CUBLAS_OP_N,
               CUBLAS_OP_N,
               m,
               n - n / 2,
               n / 2,
               &tnegone,
               W,
               cuda_data_type,
               ldw,

               work,
               cuda_data_type,
               n / 2,

               &tone,
               W + n / 2 *ldw,
               cuda_data_type,
               ldw,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  accumulatePanelQrTimeIfEnabled(&panelUpdateMilliseconds);

  return;
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

template <>
void factorPanelQr(cusolverDnHandle_t cusolver_handle,
             cublasHandle_t cublas_handle,
             long m,
             long n,
             half *A,
             long lda,
             half *W,
             long ldw,
             half *R,
             long ldr,
             half *work,
             int *info)
{

  cudaDataType_t cuda_data_type;
  cublasComputeType_t cublas_compute_type;

  cuda_data_type      = CUDA_R_16F;
  cublas_compute_type = CUBLAS_COMPUTE_16F;

  if (n <= 32)
  {
    startPanelQrTimerIfEnabled();

    factorTallSkinnyPanelQr<half, 128, 32>(cublas_handle, m, n, A, lda, R, ldr, work);


    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);

    launchIdentityMinusMatrix(gridDim, blockDim, m, n, A, lda);


    launchCopyMatrix(gridDim, blockDim, m, n, A, lda, W, ldw);


    float *_A, *_W, *_work;
    cudaMalloc((void **)&_A, sizeof(float) * m * n);
    cudaMalloc((void **)&_W, sizeof(float) * m * n);
    cudaMalloc((void **)&_work, sizeof(float) * m * n);

    matrixCpyH2F<<<gridDim, blockDim>>>(m, n, A, lda, _A, m);
    matrixCpyH2F<<<gridDim, blockDim>>>(m, n, W, ldw, _W, m);

    cusolverDnSgetrf(cusolver_handle, m, n, _A, m, _work, NULL, info);


    launchKeepLowerTriangle(gridDim, blockDim, m, n, _A, m);


    float fone = 1.0;
    cublasStrsm(cublas_handle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                m,
                n,
                &fone,
                _A,
                m,
                _W,
                m);

    matrixCpyF2H<<<gridDim, blockDim>>>(m, n, _W, m, W, ldw);
    matrixCpyF2H<<<gridDim, blockDim>>>(m, n, _A, m, A, lda);

    cudaFree(_A);
    cudaFree(_W);
    cudaFree(_work);


    accumulatePanelQrTimeIfEnabled(&panelQrMilliseconds);

    return;
  }

  factorPanelQr(cusolver_handle, cublas_handle, m, n / 2, A, lda, W, ldw, R, ldr, work, info);


  float tone    = 1.0;
  float tzero   = 0.0;
  float tnegone = -1.0;

  startPanelQrTimerIfEnabled();

  cublasGemmEx(cublas_handle,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               n / 2,
               n - n / 2,
               m,
               &tone,
               W,
               cuda_data_type,
               ldw,

               A + n / 2 * lda,
               cuda_data_type,
               lda,

               &tzero,
               work,
               cuda_data_type,
               n / 2,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);

 

  cublasGemmEx(cublas_handle,
               CUBLAS_OP_N,
               CUBLAS_OP_N,
               m,
               n - n / 2,
               n / 2,
               &tnegone,
               A,
               cuda_data_type,
               lda,

               work,
               cuda_data_type,
               n / 2,

               &tone,
               A + n / 2 * lda,
               cuda_data_type,
               lda,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  accumulatePanelQrTimeIfEnabled(&panelUpdateMilliseconds);


  dim3 gridDim((n / 2 + 32 - 1) / 32, (n - n / 2 + 32 - 1) / 32);
  dim3 blockDim(32, 32);

  launchCopyAndClear(gridDim,
                            blockDim,
                            n / 2,
                            n - n / 2,
                            A + n / 2 * lda,
                            lda,
                            R + n / 2 * ldr,
                            ldr);

  factorPanelQr(cusolver_handle,
          cublas_handle,
          m - n / 2,
          n - n / 2,
          A + n / 2 + n / 2 * lda,
          lda,
          W + n / 2 + n / 2 * ldw,
          ldw,
          R + n / 2 + n / 2 * ldr,
          ldr,
          work,
          info);


  startPanelQrTimerIfEnabled();
  cublasGemmEx(cublas_handle,
               CUBLAS_OP_T,
               CUBLAS_OP_N,
               n / 2,
               n - n / 2,
               m - n / 2,
               &tone,
               A + n / 2,
               cuda_data_type,
               lda,

               W + n / 2 + n / 2 * ldw,
               cuda_data_type,
               ldw,

               &tzero,
               work,
               cuda_data_type,
               n / 2,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  cublasGemmEx(cublas_handle,
               CUBLAS_OP_N,
               CUBLAS_OP_N,
               m,
               n - n / 2,
               n / 2,
               &tnegone,
               W,
               cuda_data_type,
               ldw,

               work,
               cuda_data_type,
               n / 2,

               &tone,
               W + n / 2 * ldw,
               cuda_data_type,
               ldw,
               cublas_compute_type,
               CUBLAS_GEMM_DEFAULT);


  accumulatePanelQrTimeIfEnabled(&panelUpdateMilliseconds);

  return;
}
