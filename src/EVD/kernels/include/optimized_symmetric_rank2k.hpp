#pragma once

#include <cublas_v2.h>

template <typename T>
void tensorCoreSymmetricRank2k(cublasHandle_t handle,
                               long n,
                               long k,
                               T alpha,
                               T* A,
                               long lda,
                               T* B,
                               long ldb,
                               T beta,
                               T* C,
                               long ldc,
                               long panel_size);
