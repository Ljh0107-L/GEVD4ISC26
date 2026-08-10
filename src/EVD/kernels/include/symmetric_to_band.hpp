#pragma once

#include <cusolverDn.h>

template <typename T>
void reduceSymmetricMatrixToBand(
    cusolverDnHandle_t cusolver_handle,
    cublasHandle_t cublas_handle,
    long rows,
    long columns,
    long block_size,
    long panel_size,
    T* original_matrix,
    long original_leading_dimension,
    T* panel_matrix,
    long panel_leading_dimension,
    T* W,
    long leading_dimension_W,
    T* Y,
    long leading_dimension_Y,
    T* Z,
    long leading_dimension_Z,
    T* R,
    long leading_dimension_R,
    T* workspace,
    int* numerical_info);
