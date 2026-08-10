
#include "blocked_wy.hpp"
#include "runtime/error_handling.hpp"

void accumulateBlockedWY(cublasHandle_t cublas_handle,
                         long rows,
                         long columns,
                         long panel_width,
                         double* W,
                         long leading_dimension_W,
                         double* Y,
                         long leading_dimension_Y,
                         double* workspace)
{
  const double one = 1.0;
  const double zero = 0.0;
  const double negative_one = -1.0;
  const long workspace_leading_dimension = rows;

  for (long column = 2 * panel_width;
       column <= columns;
       column += panel_width) {
    CUBLAS_CHECK_LOCAL(cublasDgemm(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        column - panel_width,
        panel_width,
        rows,
        &one,
        Y,
        leading_dimension_Y,
        W + (column - panel_width) * leading_dimension_W,
        leading_dimension_W,
        &zero,
        workspace,
        workspace_leading_dimension));

    CUBLAS_CHECK_LOCAL(cublasDgemm(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        rows,
        panel_width,
        column - panel_width,
        &negative_one,
        W,
        leading_dimension_W,
        workspace,
        workspace_leading_dimension,
        &one,
        W + (column - panel_width) * leading_dimension_W,
        leading_dimension_W));
  }
}
