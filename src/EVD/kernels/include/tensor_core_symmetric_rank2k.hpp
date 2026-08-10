#pragma once

#include <cusolverDn.h>


void tensorCoreSymmetricRank2k(cublasHandle_t handle,
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
              __half *hwork,
              long int nb);
