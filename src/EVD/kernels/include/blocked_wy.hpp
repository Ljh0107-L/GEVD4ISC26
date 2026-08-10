#pragma once

#include <cublas_v2.h>

void accumulateBlockedWY(cublasHandle_t cublas_handle,
                         long rows,
                         long columns,
                         long panel_width,
                         double* W,
                         long leading_dimension_W,
                         double* Y,
                         long leading_dimension_Y,
                         double* workspace);
