
#include "orthogonal_from_wy.hpp"
#include "runtime/error_handling.hpp"
#include "utility_kernels.cuh"

void formOrthogonalMatrixFromWY(cublasHandle_t cublas_handle,
                                long rows,
                                long reflector_count,
                                double* orthogonal_matrix,
                                long orthogonal_leading_dimension,
                                double* W,
                                long leading_dimension_W,
                                double* Y,
                                long leading_dimension_Y)
{
  const double one = 1.0;
  const double zero = 0.0;

  CUBLAS_CHECK_LOCAL(cublasDgemm(cublas_handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_T,
                                 rows,
                                 rows,
                                 reflector_count,
                                 &one,
                                 W,
                                 leading_dimension_W,
                                 Y,
                                 leading_dimension_Y,
                                 &zero,
                                 orthogonal_matrix,
                                 orthogonal_leading_dimension));

  const dim3 block(32, 32);
  const dim3 grid((rows + 31) / 32, (rows + 31) / 32);
  launchIdentityMinusMatrix(grid,
                       block,
                       rows,
                       rows,
                       orthogonal_matrix,
                       orthogonal_leading_dimension);
}
