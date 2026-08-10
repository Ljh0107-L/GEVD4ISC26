set(_standard_evd_private_includes
  ${PROJECT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include)

# Keep the compile-time BC-back tuning hooks from the source implementation.
# Empty values select the kernel defaults (8, 24, and 90).
set(EVD_BC_BACK_U_COUNT "" CACHE STRING
    "Override BC-back register rows per thread")
set(EVD_BC_BACK_MAX_WARP_COUNT "" CACHE STRING
    "Override BC-back warp groups per block")
set(EVD_BC_BACK_U_COL_EXTERN_COUNT "" CACHE STRING
    "Override BC-back U column tile")

# Keep the optimized standard-EVD engine in three focused libraries: utility
# kernels, numerical kernels, and the distributed stage driver.
add_library(gevd4isc26_evd_utils SHARED
  runtime/error_handling.cpp
  kernels/common/utility_kernels.cu)
target_include_directories(gevd4isc26_evd_utils PRIVATE
  ${_standard_evd_private_includes}
  ${NCCL_INCLUDE_DIR}
  ${GEVD_MPI_INCLUDE_DIRS})
target_link_libraries(gevd4isc26_evd_utils PRIVATE
  MPI::MPI_CXX
  CUDA::cudart CUDA::cublas CUDA::curand)

add_library(gevd4isc26_evd_kernels SHARED
  kernels/backtransform/bc_backtransform.cu
  kernels/sbr/blocked_wy.cu
  kernels/sbr/orthogonal_from_wy.cu
  kernels/sbr/panel_qr.cu
  kernels/sbr/symmetric_to_band.cu
  kernels/tensor/matrix_partition.cpp
  kernels/tensor/optimized_symmetric_rank2k.cu
  kernels/tensor/tensor_core_symmetric_rank2k.cu)
target_include_directories(gevd4isc26_evd_kernels PRIVATE
  ${_standard_evd_private_includes}
  ${NCCL_INCLUDE_DIR}
  ${GEVD_MPI_INCLUDE_DIRS})
target_link_libraries(gevd4isc26_evd_kernels PRIVATE
  gevd4isc26_evd_utils
  MPI::MPI_CXX
  ${NCCL_LIBRARY}
  CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::curand)
if(NOT "${EVD_BC_BACK_U_COUNT}" STREQUAL "")
  target_compile_definitions(gevd4isc26_evd_kernels PRIVATE
    EVD_BC_BACK_ROWS_PER_THREAD=${EVD_BC_BACK_U_COUNT})
endif()
if(NOT "${EVD_BC_BACK_MAX_WARP_COUNT}" STREQUAL "")
  target_compile_definitions(gevd4isc26_evd_kernels PRIVATE
    EVD_BC_BACK_MAX_WARP_GROUPS=${EVD_BC_BACK_MAX_WARP_COUNT})
endif()
if(NOT "${EVD_BC_BACK_U_COL_EXTERN_COUNT}" STREQUAL "")
  target_compile_definitions(gevd4isc26_evd_kernels PRIVATE
    EVD_BC_BACK_COLUMN_TILE=${EVD_BC_BACK_U_COL_EXTERN_COUNT})
endif()

add_library(gevd4isc26_standard_evd_engine SHARED
  engine/evd_engine.cu)
target_compile_definitions(gevd4isc26_standard_evd_engine PRIVATE MKL_ILP64)
target_include_directories(gevd4isc26_standard_evd_engine PRIVATE
  ${_standard_evd_private_includes}
  ${NCCL_INCLUDE_DIR}
  ${MKL_INCLUDE_DIR}
  ${GEVD_MPI_INCLUDE_DIRS})
target_link_libraries(gevd4isc26_standard_evd_engine PRIVATE
  gevd4isc26_d_and_c
  gevd4isc26_evd_kernels
  gevd4isc26_evd_utils
  MPI::MPI_CXX
  ${NCCL_LIBRARY}
  # Debug-only dense reference checks in diagnostics/evd_verification.cuh
  # still use MKL. The production tridiagonal D&C path is owned by the
  # independent gevd4isc26_d_and_c target above.
  ${MKL_GF_ILP64_LIBRARY}
  ${MKL_GNU_THREAD_LIBRARY}
  ${MKL_CORE_LIBRARY}
  ${GOMP_LIBRARY}
  CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::curand
  ${CMAKE_DL_LIBS} m)
if(NOT "${EVD_BC_BACK_U_COUNT}" STREQUAL "")
  target_compile_definitions(gevd4isc26_standard_evd_engine PRIVATE
    EVD_BC_BACK_ROWS_PER_THREAD=${EVD_BC_BACK_U_COUNT})
endif()
if(NOT "${EVD_BC_BACK_U_COL_EXTERN_COUNT}" STREQUAL "")
  target_compile_definitions(gevd4isc26_standard_evd_engine PRIVATE
    EVD_BC_BACK_COLUMN_TILE=${EVD_BC_BACK_U_COL_EXTERN_COUNT})
endif()

foreach(_target IN ITEMS
    gevd4isc26_evd_utils
    gevd4isc26_evd_kernels
    gevd4isc26_standard_evd_engine)
  set_target_properties(${_target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    BUILD_RPATH "${GEVD_RUNTIME_DIRECTORIES};$ORIGIN"
    INSTALL_RPATH "$ORIGIN;${GEVD_RUNTIME_DIRECTORIES}")
  target_compile_options(${_target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endforeach()

set(GEVD4ISC26_STANDARD_EVD_TARGETS
  gevd4isc26_evd_utils
  gevd4isc26_evd_kernels
  gevd4isc26_standard_evd_engine
  PARENT_SCOPE)
