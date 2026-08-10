
#pragma once

#include <cusolverDn.h>


template <typename T>
void factorPanelQr(cusolverDnHandle_t cusolver_handle,
                   cublasHandle_t cublas_handle,
                   long rows,
                   long columns,
                   T* panel,
                   long panel_leading_dimension,
                   T* W,
                   long leading_dimension_W,
                   T* R,
                   long leading_dimension_R,
                   T* workspace,
                   int* numerical_info);
