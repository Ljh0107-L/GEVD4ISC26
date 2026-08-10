#include <iostream>
#include <string>
#include <vector>

#include "blocked_wy.hpp"
#include "optimized_symmetric_rank2k.hpp"
#include "orthogonal_from_wy.hpp"
#include "panel_qr.hpp"
#include "runtime/cuda_check.hpp"
#include "tall_skinny_qr.hpp"
#include "utility_kernels.cuh"
#include <assert.h>

namespace {

float panelQrMilliseconds = 0.0F;
float symmetricUpdateMilliseconds = 0.0F;
float gemmMilliseconds = 0.0F;

}  // namespace



template <typename T>
void reduceSymmetricMatrixToBand(cusolverDnHandle_t cusolver_handle,
                  cublasHandle_t cublas_handle,
                  long M,
                  long N,
                  long b,
                  long nb,
                  T *dOriA,
                  long ldOriA,
                  T *dA,
                  long ldA,
                  T *dW,
                  long ldW,
                  T *dY,
                  long ldY,
                  T *dZ,
                  long ldZ,
                  T *dR,
                  long ldR,
                  T *work,
                  int *info)
{
  if (0 >= M)
  {
    return;
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


  T done     = 1.0;
  T dzero    = 0.0;
  T dnegone  = -1.0;
  T dneghalf = -0.5;

  T *OA = dOriA + b * ldOriA + b;

  long ldWork = ldOriA;

  long i;
  for (i = b; (i <= nb) && (i < N); i += b)
  {
    long m = M - i;
    long n = b;

    T *dPanel = dA + i + (i - b) * ldA;

    T *dPanelW = dW + i + (i - b) * ldW;

    T *dPanelR = dR + i + (i - b) * ldR;


    startTimer();
    factorPanelQr(cusolver_handle,
            cublas_handle,
            m,
            n,
            dPanel,
            ldA,
            dPanelW,
            ldW,
            dPanelR,
            ldR,
            work,
            info);

    panelQrMilliseconds += stopTimer();

    EVD_CUDA_CHECK(cudaGetLastError());
    EVD_CUDA_CHECK(cudaDeviceSynchronize());


    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);

    T *dPanelY = dY + i + (i - b) * ldY;
    launchCopyMatrix(gridDim, blockDim, m, n, dPanel, ldA, dPanelY, ldY);

    launchExtractHouseholderVectors(gridDim, blockDim, m, n, dPanelR, ldR, dPanel, ldA);


    T *dPanelZ = dZ + i + (i - b) * ldZ;

    startTimer();
    if (i == b)
    {

      cublasGemmEx(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   m,
                   b,
                   m,
                   &done,
                   OA,
                   cuda_data_type,
                   ldOriA,

                   dPanelW,
                   cuda_data_type,
                   ldW,

                   &dzero,
                   dPanelZ,
                   cuda_data_type,
                   ldZ,

                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT);


      cublasGemmEx(cublas_handle,
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   b,
                   b,
                   m,
                   &done,
                   dPanelW,
                   cuda_data_type,
                   ldW,
                   dPanelZ,
                   cuda_data_type,
                   ldZ,
                   &dzero,
                   work,
                   cuda_data_type,
                   ldWork,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT);


      cublasGemmEx(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   m,
                   b,
                   b,
                   &dneghalf,
                   dPanelY,
                   cuda_data_type,
                   ldY,
                   work,
                   cuda_data_type,
                   ldWork,
                   &done,
                   dPanelZ,
                   cuda_data_type,
                   ldZ,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT);
    }
    else
    {
      
      cublasGemmEx(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   m,
                   b,
                   m,
                   &done,
                   OA + (i - b) + (i - b) * ldOriA,
                   cuda_data_type,
                   ldOriA,
                   dPanelW,
                   cuda_data_type,
                   ldW,
                   &dzero,
                   dPanelZ,
                   cuda_data_type,
                   ldZ,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT);


      cublasGemmEx(cublas_handle,
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   i - b,
                   b,
                   m,
                   &done,
                   dZ + i,
                   cuda_data_type,
                   ldZ,
                   dPanelW,
                   cuda_data_type,
                   ldW,
                   &dzero,
                   work,
                   cuda_data_type,
                   ldWork,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT);

 

      cublasGemmEx(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   m,
                   b,
                   i - b,
                   &dnegone,
                   dY + i,
                   cuda_data_type,
                   ldY,
                   work,
                   cuda_data_type,
                   ldWork,
                   &done,
                   dPanelZ,
                   cuda_data_type,
                   ldZ,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT);


      cublasGemmEx(cublas_handle,
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   i - b,
                   b,
                   m,
                   &done,
                   dY + i,
                   cuda_data_type,
                   ldY,
                   dPanelW,
                   cuda_data_type,
                   ldW,
                   &dzero,
                   work,
                   cuda_data_type,
                   ldWork,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT);

      cublasGemmEx(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   m,
                   b,
                   i - b,
                   &dnegone,
                   dZ + i,
                   cuda_data_type,
                   ldZ,
                   work,
                   cuda_data_type,
                   ldWork,
                   &done,
                   dPanelZ,
                   cuda_data_type,
                   ldZ,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT);


      cublasGemmEx(cublas_handle,
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   b,
                   b,
                   m,
                   &done,
                   dPanelW,
                   cuda_data_type,
                   ldW,
                   dPanelZ,
                   cuda_data_type,
                   ldZ,
                   &dzero,
                   work,
                   cuda_data_type,
                   ldWork,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT);

 

      cublasGemmEx(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   m,
                   b,
                   b,
                   &dneghalf,
                   dPanelY,
                   cuda_data_type,
                   ldY,
                   work,
                   cuda_data_type,
                   ldWork,
                   &done,
                   dPanelZ,
                   cuda_data_type,
                   ldZ,
                   cublas_compute_type,
                   CUBLAS_GEMM_DEFAULT);
    }


    if (i < nb)
    {
     
      cublasGemmEx(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   m,
                   b,
                   i,
                   &dnegone,
                   dY + i,
                   cuda_data_type,
                   ldY,
                   dZ + i,
                   cuda_data_type,
                   ldZ,
                   &done,
                   dA + i + i * ldA,
                   cuda_data_type,
                   ldA,
                   cuda_data_type,
                   CUBLAS_GEMM_DEFAULT);

 

      cublasGemmEx(cublas_handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   m,
                   b,
                   i,
                   &dnegone,
                   dZ + i,
                   cuda_data_type,
                   ldZ,
                   dY + i,
                   cuda_data_type,
                   ldY,
                   &done,
                   dA + i + i * ldA,
                   cuda_data_type,
                   ldA,
                   cuda_data_type,
                   CUBLAS_GEMM_DEFAULT);
    }

    gemmMilliseconds += stopTimer();


  }



  if (0 >= N - nb)
  {
    return;
  }



  long lm = M - nb; 
  long ln = nb;

  dim3 block1(32, 32);
  dim3 grid1((lm + 31) / 32, (lm + 31) / 32);

  startTimer();

  tensorCoreSymmetricRank2k(cublas_handle,
                  lm,
                  ln,
                  dnegone,
                  dY + nb,
                  ldY,
                  dZ + nb,
                  ldZ,
                  done,
                  OA + (nb - b) + (nb - b) * ldOriA,
                  ldOriA,
                  nb);
  symmetricUpdateMilliseconds += stopTimer();

  launchCopyLowerToUpper(grid1, block1, lm, OA + (nb - b) + (nb - b) * ldOriA, ldOriA);

 
  dim3 gridDim2((M - nb + 31) / 32, (N - nb + 31) / 32);
  dim3 blockDim2(32, 32);


  launchCopyMatrix(gridDim2,
                          blockDim2,
                          M - nb,
                          N - nb,
                          dOriA + nb + nb * ldOriA,
                          ldOriA,
                          dA + nb + nb * ldA,
                          ldA);

  M = N = M - nb;
  dA    = dA + nb + nb * ldA;
  dW = dW + nb + nb * ldW;
  dY = dY + nb + nb * ldY;

  lm = M - b;
  dim3 grid2((lm + 31) / 32, (ln + 31) / 32);

  launchClearMatrix(grid2, block1, lm, ln, dZ + b, ldZ);

  dR = dR + nb;

  dOriA = dOriA + nb + nb * ldOriA;

  reduceSymmetricMatrixToBand(cusolver_handle,
               cublas_handle,
               M,
               N,
               b,
               nb,
               dOriA,
               ldOriA,
               dA,
               ldA,
               dW,
               ldW,
               dY,
               ldY,
               dZ,
               ldZ,
               dR,
               ldR,
               work,
               info);
}

template void reduceSymmetricMatrixToBand(cusolverDnHandle_t cusolver_handle,
                  cublasHandle_t cublas_handle,
                  long M,
                  long N,
                  long b,
                  long nb,
                  double *dOriA,
                  long ldOriA,
                  double *dA,
                  long ldA,
                  double *dW,
                  long ldW,
                  double *dY,
                  long ldY,
                  double *dZ,
                  long ldZ,
                  double *dR,
                  long ldR,
                  double *work,
                  int *info);
