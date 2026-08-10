#pragma once



template <typename T>
void launchClearMatrix(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, cudaStream_t stream = NULL);



template <typename T>
void launchSetTriangularValue(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, T v, cudaStream_t stream = NULL);


template <typename T>
void launchCopyLowerToUpper(dim3 gridDim, dim3 blockDim, long n, T *A, long ldA);


template <typename T>
void launchCopyAndClear(dim3 gridDim,
                               dim3 blockDim,
                               long m,
                               long n,
                               T *srcM,
                               long lds,
                               T *dstM,
                               long ldd);



template <typename T>
void launchIdentityMinusMatrix(dim3 gridDim, dim3 blockDim, long m, long n, T *Q, long ldq);

void launchMatrixDifference(dim3 gridDim,
                          dim3 blockDim,
                          long m,
                          long n,
                          double *A,
                          long ldA,
                          double *B,
                          long ldB);

void launchAbsoluteMatrixDifference(dim3 gridDim,
                                dim3 blockDim,
                                long m,
                                long n,
                                double *A,
                                long ldA,
                                double *B,
                                long ldB);



template <typename T>
void launchCopyMatrix(dim3 gridDim,
                             dim3 blockDim,
                             long m,
                             long n,
                             T *srcM,
                             long lds,
                             T *dstM,
                             long ldd);


template <typename T>
void launchTransposeMatrix(dim3 gridDim,
                                      dim3 blockDim,
                                      long m,
                                      long n,
                                      T *srcM,
                                      long lds,
                                      T *dstM,
                                      long ldd);

template <typename T>
void launchExtractHouseholderVectors(dim3 gridDim, dim3 blockDim, int m, int n, T *A, int ldA, T *U, int ldU);



template <typename T>
void launchKeepLowerTriangle(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA);



template <typename T>
void launchPackLowerTriangle(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, T *B, cudaStream_t stream = NULL);

template <typename T>
void launchUnpackLowerTriangle(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, T *B, cudaStream_t stream = NULL);


void launchScaleMatrix(dim3 gridDim,
                               dim3 blockDim,
                               long m,
                               long n,
                               double *A,
                               long ldA,
                               double scaler);

double findVectorAbsMax(double *d_array, int n);
