add_subdirectory(d_and_c)
add_subdirectory(evd)

add_library(gevd4isc26 SHARED
  diagnostics/timing.cpp
  diagnostics/validation.cpp
  distributed/context.cu
  distributed/distributed_matrix.cpp
  evd/adapter.cu
  gevd/backtransform.cu
  gevd/cholesky.cu
  gevd/factor_cache.cpp
  gevd/numerical_info.cu
  gevd/pipeline.cu
  gevd/standard_reduction.cu
  gevd/workspace.cpp
  io/matrix_io.cpp
  runtime/gpu_resources.cu
  runtime/mpi_communicator.cpp)
add_library(GEVD4ISC26::gevd4isc26 ALIAS gevd4isc26)

target_include_directories(gevd4isc26
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUSOLVERMP_INCLUDE_DIR}
    ${CUBLASMP_INCLUDE_DIR}
    ${NCCL_INCLUDE_DIR}
    ${GEVD_MPI_INCLUDE_DIRS})

# cuBLASMp 0.5 carries an $ORIGIN-only RUNPATH.  Retain cuBLAS and NVSHMEM as
# earlier direct dependencies so the loader has already resolved them before
# it visits cuBLASMp's transitive dependency list.
target_link_libraries(gevd4isc26
  PUBLIC MPI::MPI_CXX
  PRIVATE
    ${CUSOLVERMP_LIBRARY}
    "-Wl,--no-as-needed"
    CUDA::cublas
    ${NVSHMEM_HOST_LIBRARY}
    "-Wl,--as-needed"
    ${CUBLASMP_LIBRARY}
    ${NCCL_LIBRARY}
    gevd4isc26_standard_evd_engine
    CUDA::cudart
    CUDA::cusolver
    CUDA::curand
    Threads::Threads
    ${CMAKE_DL_LIBS}
    m)

target_compile_options(gevd4isc26 PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

# cuBLASMp and cuSOLVERMp have CUDA/NVSHMEM dependencies of their own.  Use a
# transitive ELF RPATH so those libraries are resolved from the exact toolkit
# selected at configure time, without requiring a broad LD_LIBRARY_PATH.
target_link_options(gevd4isc26 PRIVATE
  "LINKER:--disable-new-dtags")
foreach(_runtime_directory IN LISTS GEVD_RUNTIME_DIRECTORIES)
  target_link_options(gevd4isc26 INTERFACE
    "LINKER:-rpath-link,${_runtime_directory}")
endforeach()

set_target_properties(gevd4isc26 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "${GEVD_RUNTIME_DIRECTORIES}"
  INSTALL_RPATH "$ORIGIN;${GEVD_RUNTIME_DIRECTORIES}")

set(GEVD4ISC26_INTERNAL_TARGETS
  ${GEVD4ISC26_D_AND_C_TARGETS}
  ${GEVD4ISC26_STANDARD_EVD_TARGETS}
  PARENT_SCOPE)
