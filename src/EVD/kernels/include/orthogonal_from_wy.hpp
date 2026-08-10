#pragma once

#include <cublas_v2.h>

void formOrthogonalMatrixFromWY(cublasHandle_t cublas_handle,
                                long rows,
                                long reflector_count,
                                double* orthogonal_matrix,
                                long orthogonal_leading_dimension,
                                double* W,
                                long leading_dimension_W,
                                double* Y,
                                long leading_dimension_Y);
