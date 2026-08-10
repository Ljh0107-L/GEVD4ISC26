#include <cuda_fp16.h>

#include "utility_kernels.cuh"
#include <stdio.h>

template <typename T>
__global__ void clearMatrixKernel(long m, long n, T *A, long ldA)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < m && j < n)
  {
    A[i + j * ldA] = 0.0;
  }
}

template <typename T>
void launchClearMatrix(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, cudaStream_t stream)
{
  clearMatrixKernel<<<gridDim, blockDim,0, stream>>>(m, n, A, ldA);
}

template void
launchClearMatrix(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA, cudaStream_t stream);

template void
launchClearMatrix(dim3 gridDim, dim3 blockDim, long m, long n, float *A, long ldA, cudaStream_t stream);

template void
launchClearMatrix(dim3 gridDim, dim3 blockDim, long m, long n, half *A, long ldA, cudaStream_t stream);

template <typename T>
static __global__ void setTriangularValueKernel(long m, long n, T *A, long ldA, T v)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;


  if ((i < m) && (i < n))
  {
    A[i + i * ldA] = v;
  }
}

template <typename T>
void launchSetTriangularValue(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, T v, cudaStream_t stream)
{
  setTriangularValueKernel<<<gridDim, blockDim, 0, stream>>>(m, n, A, ldA, v);
}

template void launchSetTriangularValue(dim3 gridDim,
                                            dim3 blockDim,
                                            long m,
                                            long n,
                                            double *A,
                                            long ldA,
                                            double v, cudaStream_t stream);
template void launchSetTriangularValue(dim3 gridDim,
                                            dim3 blockDim,
                                            long m,
                                            long n,
                                            float *A,
                                            long ldA,
                                            float v, cudaStream_t stream);
template void launchSetTriangularValue(dim3 gridDim,
                                            dim3 blockDim,
                                            long m,
                                            long n,
                                            half *A,
                                            long ldA,
                                            half v, cudaStream_t stream);

template <typename T>
__global__ void copyLowerToUpperKernel(long n, T *A, long ldA)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;



  if (i < n && j < n)
  {
    if (j > i)
      A[i + j * ldA] = A[j + i * ldA];
  }
}

template <typename T>
void launchCopyLowerToUpper(dim3 gridDim, dim3 blockDim, long n, T *A, long ldA)
{
  copyLowerToUpperKernel<<<gridDim, blockDim>>>(n, A, ldA);
}

template void launchCopyLowerToUpper(dim3 gridDim, dim3 blockDim, long n, double *A, long ldA);
template void launchCopyLowerToUpper(dim3 gridDim, dim3 blockDim, long n, float *A, long ldA);
template void launchCopyLowerToUpper(dim3 gridDim, dim3 blockDim, long n, half *A, long ldA);

template <typename T>
__global__ void copyAndClearKernel(long m, long n, T *srcM, long lds, T *dstM, long ldd)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < m && j < n)
  {
    dstM[i + j * ldd] = srcM[i + j * lds];
    srcM[i + j * lds] = 0.0;
  }
}

template <typename T>
void launchCopyAndClear(dim3 gridDim,
                               dim3 blockDim,
                               long m,
                               long n,
                               T *srcM,
                               long lds,
                               T *dstM,
                               long ldd)
{
  copyAndClearKernel<<<gridDim, blockDim>>>(m, n, srcM, lds, dstM, ldd);
}

template void launchCopyAndClear(dim3 gridDim,
                                        dim3 blockDim,
                                        long m,
                                        long n,
                                        double *srcM,
                                        long lds,
                                        double *dstM,
                                        long ldd);

template void launchCopyAndClear(dim3 gridDim,
                                        dim3 blockDim,
                                        long m,
                                        long n,
                                        float *srcM,
                                        long lds,
                                        float *dstM,
                                        long ldd);

template void launchCopyAndClear(dim3 gridDim,
                                        dim3 blockDim,
                                        long m,
                                        long n,
                                        half *srcM,
                                        long lds,
                                        half *dstM,
                                        long ldd);

template <typename T>
__global__ void identityMinusMatrixKernel(long m, long n, T *Q, long ldq)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // printf("come %d, %d, %d,\n", __LINE__, i, j);
  // __syncthreads();

  if (i < m && j < n)
  {
    if (i == j)
    {
      Q[i + j * ldq] = (T)1.0 - Q[i + j * ldq];
    }
    else
    {
      Q[i + j * ldq] = -Q[i + j * ldq];
    }


  }
}

template <typename T>
void launchIdentityMinusMatrix(dim3 gridDim, dim3 blockDim, long m, long n, T *Q, long ldq)
{
  identityMinusMatrixKernel<<<gridDim, blockDim>>>(m, n, Q, ldq);
}

template void launchIdentityMinusMatrix(dim3 gridDim, dim3 blockDim, long m, long n, double *Q, long ldq);

template void launchIdentityMinusMatrix(dim3 gridDim, dim3 blockDim, long m, long n, float *Q, long ldq);

template void launchIdentityMinusMatrix(dim3 gridDim, dim3 blockDim, long m, long n, half *Q, long ldq);

__global__ void matrixDifferenceKernel(long m, long n, double *A, long ldA, double *B, long ldB)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // printf("come %d, %d, %d,\n", __LINE__, i, j);
  // __syncthreads();

  if (i < m && j < n)
  {

    A[i + j * ldA] -= B[i + j * ldB];

  }
}

void launchMatrixDifference(dim3 gridDim,
                          dim3 blockDim,
                          long m,
                          long n,
                          double *A,
                          long ldA,
                          double *B,
                          long ldB)
{
  matrixDifferenceKernel<<<gridDim, blockDim>>>(m, n, A, ldA, B, ldB);
}

__global__ void absoluteMatrixDifferenceKernel(long m, long n, double *A, long ldA, double *B, long ldB)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  // printf("come %d, %d, %d,\n", __LINE__, i, j);
  // __syncthreads();

  if (i < m && j < n)
  {
    // A[i + j * ldA] = abs(A[i + j * ldA]);
    double t = abs(A[i + j * ldA]);

    A[i + j * ldA] = t - abs(B[i + j * ldB]);

  }
}

void launchAbsoluteMatrixDifference(dim3 gridDim,
                                dim3 blockDim,
                                long m,
                                long n,
                                double *A,
                                long ldA,
                                double *B,
                                long ldB)
{
  absoluteMatrixDifferenceKernel<<<gridDim, blockDim>>>(m, n, A, ldA, B, ldB);
}

template <typename T>
__global__ void copyMatrixKernel(long m, long n, T *srcM, long lds, T *dstM, long ldd)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < m && j < n)
  {
    dstM[i + j * ldd] = srcM[i + j * lds];
  }
}

template <typename T>
void launchCopyMatrix(dim3 gridDim,
                             dim3 blockDim,
                             long m,
                             long n,
                             T *srcM,
                             long lds,
                             T *dstM,
                             long ldd)
{
  copyMatrixKernel<<<gridDim, blockDim>>>(m, n, srcM, lds, dstM, ldd);
}

template void launchCopyMatrix(dim3 gridDim,
                                      dim3 blockDim,
                                      long m,
                                      long n,
                                      double *srcM,
                                      long lds,
                                      double *dstM,
                                      long ldd);
template void launchCopyMatrix(dim3 gridDim,
                                      dim3 blockDim,
                                      long m,
                                      long n,
                                      float *srcM,
                                      long lds,
                                      float *dstM,
                                      long ldd);

template void launchCopyMatrix(dim3 gridDim,
                                      dim3 blockDim,
                                      long m,
                                      long n,
                                      half *srcM,
                                      long lds,
                                      half *dstM,
                                      long ldd);

template <typename T>
__global__ void transposeMatrixKernel(long m, long n, T *srcM, long lds, T *dstM, long ldd)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < m && j < n)
  {
    dstM[j + i * ldd] = srcM[i + j * lds];
  }
}

template <typename T>
void launchTransposeMatrix(dim3 gridDim,
                                      dim3 blockDim,
                                      long m,
                                      long n,
                                      T *srcM,
                                      long lds,
                                      T *dstM,
                                      long ldd)
{
  transposeMatrixKernel<<<gridDim, blockDim>>>(m, n, srcM, lds, dstM, ldd);
}

template void launchTransposeMatrix(dim3 gridDim,
                                               dim3 blockDim,
                                               long m,
                                               long n,
                                               double *srcM,
                                               long lds,
                                               double *dstM,
                                               long ldd);

template void launchTransposeMatrix(dim3 gridDim,
                                               dim3 blockDim,
                                               long m,
                                               long n,
                                               float *srcM,
                                               long lds,
                                               float *dstM,
                                               long ldd);

template <typename T>
__global__ void extractHouseholderVectorsKernel(int m, int n, T *A, int ldA, T *U, int ldU)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < n)
  {
    if (i > j)
      U[i + j * ldU] = 0;
    else
      U[i + j * ldU] = A[i + j * ldA];
  }
}

template <typename T>
void launchExtractHouseholderVectors(dim3 gridDim, dim3 blockDim, int m, int n, T *A, int ldA, T *U, int ldU)
{
  extractHouseholderVectorsKernel<<<gridDim, blockDim>>>(m, n, A, ldA, U, ldU);
}

template void
launchExtractHouseholderVectors(dim3 gridDim, dim3 blockDim, int m, int n, double *A, int ldA, double *U, int ldU);

template void
launchExtractHouseholderVectors(dim3 gridDim, dim3 blockDim, int m, int n, float *A, int ldA, float *U, int ldU);

template void
launchExtractHouseholderVectors(dim3 gridDim, dim3 blockDim, int m, int n, half *A, int ldA, half *U, int ldU);

template <typename T>
__global__ void keepLowerTriangleKernel(long m, long n, T *dA, long lda)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;


  if (i < m && j < n)
  {
    if (i < j)
    {
      dA[i + j * lda] = 0.0;
    }
    else if (i == j)
    {
      dA[i + j * lda] = 1.0;
    }

  }
}

template <typename T>
void launchKeepLowerTriangle(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA)
{
  keepLowerTriangleKernel<<<gridDim, blockDim>>>(m, n, A, ldA);
}

template void
launchKeepLowerTriangle(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA);
template void launchKeepLowerTriangle(dim3 gridDim, dim3 blockDim, long m, long n, float *A, long ldA);

template <typename T>
__global__ void packLowerTriangleKernel(long m, long n, T *A, long ldA, T *B)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if ((i < m) && (i < n))
  {
    B[i] = A[i + i * ldA];
  }
}

template <typename T>
void launchPackLowerTriangle(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, T *B, cudaStream_t stream)
{
  packLowerTriangleKernel<<<gridDim, blockDim, 0, stream>>>(m, n, A, ldA, B);
}
template void launchPackLowerTriangle(dim3 gridDim,
                                          dim3 blockDim,
                                          long m,
                                          long n,
                                          double *A,
                                          long ldA,
                                          double *B,
                                        cudaStream_t stream);

template void launchPackLowerTriangle(dim3 gridDim,
                                          dim3 blockDim,
                                          long m,
                                          long n,
                                          float *A,
                                          long ldA,
                                          float *B,
                                        cudaStream_t stream);

template void launchPackLowerTriangle(dim3 gridDim,
                                          dim3 blockDim,
                                          long m,
                                          long n,
                                          half *A,
                                          long ldA,
                                          half *B,
                                        cudaStream_t stream);


template <typename T>
__global__ void unpackLowerTriangleKernel(long m, long n, T *A, long ldA, T *B)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if ((i < m) && (i < n))
  {
    A[i + i * ldA] = B[i];
  }
}

template <typename T>
void launchUnpackLowerTriangle(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, T *B, cudaStream_t stream)
{
  unpackLowerTriangleKernel<<<gridDim, blockDim, 0, stream>>>(m, n, A, ldA, B);
}
template void launchUnpackLowerTriangle(dim3 gridDim,
                                          dim3 blockDim,
                                          long m,
                                          long n,
                                          double *A,
                                          long ldA,
                                          double *B,
                                        cudaStream_t stream);

template void launchUnpackLowerTriangle(dim3 gridDim,
                                          dim3 blockDim,
                                          long m,
                                          long n,
                                          float *A,
                                          long ldA,
                                          float *B,
                                        cudaStream_t stream);

template void launchUnpackLowerTriangle(dim3 gridDim,
                                          dim3 blockDim,
                                          long m,
                                          long n,
                                          half *A,
                                          long ldA,
                                          half *B,
                                        cudaStream_t stream);

__global__ void scaleMatrixKernel(long m, long n, double *A, long ldA, double scaler)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;


  if (i < m && j < n)
  {

    A[i + j * ldA] *= scaler;

    // printf("come %d, %d, %d,\n", __LINE__, i, j);
    // __syncthreads();
  }
}

void launchScaleMatrix(dim3 gridDim,
                               dim3 blockDim,
                               long m,
                               long n,
                               double *A,
                               long ldA,
                               double scaler)
{
  scaleMatrixKernel<<<gridDim, blockDim>>>(m, n, A, ldA, scaler);
}

__global__ void findAbsMaxKernel(double *d_array, double *d_max, int n)
{
  extern __shared__ double sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  // Load data into shared memory
  if (idx < n)
  {
    sdata[tid] = abs(d_array[idx]);
  }
  else
  {
    // sdata[tid] = -INFINITY; // Ensure out of bounds values do not affect max
    sdata[tid] = 0;
  }
  __syncthreads();

  // Perform reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      sdata[tid] = max(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  // Write the maximum value for this block to the output array
  if (tid == 0)
  {
    d_max[blockIdx.x] = sdata[0];
  }
}

double findVectorAbsMax(double *d_array, int n)
{
  double *d_max;
  double *h_max       = new double[(n + 255) / 256];
  int threadsPerBlock = 256;
  int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

  cudaMalloc((void **)&d_max, blocksPerGrid * sizeof(double));

  findAbsMaxKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_array,
                                                                                         d_max,
                                                                                         n);

  cudaMemcpy(h_max, d_max, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

  double max_val = 0;
  for (int i = 0; i < blocksPerGrid; i++)
  {
    max_val = std::max(max_val, h_max[i]);
  }

  cudaFree(d_max);
  delete[] h_max;

  return max_val;
}
