# Central dependency discovery for the GEVD library and its bundled EVD
# engine. This module deliberately exports resolved paths as directory
# variables; numerical targets remain responsible for their own link scope.

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Conda CUDA packages expose convenience symlinks from their top-level lib
# directory, which may also contain a different MPI implementation.  Point
# CMake's imported CUDA targets at the real files under the toolkit target
# directory so the generated runpaths never need that mixed directory.
foreach(_cuda_target IN ITEMS
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cusolver
    CUDA::curand
    CUDA::cusparse
    CUDA::nvJitLink)
  if(TARGET "${_cuda_target}")
    get_target_property(_cuda_imported_location
      "${_cuda_target}" IMPORTED_LOCATION)
    if(_cuda_imported_location AND EXISTS "${_cuda_imported_location}")
      file(REAL_PATH "${_cuda_imported_location}" _cuda_real_location)
      set_property(TARGET "${_cuda_target}" PROPERTY
        IMPORTED_LOCATION "${_cuda_real_location}")
    endif()
  endif()
endforeach()

# When mpicxx is selected as CMAKE_CXX_COMPILER, CMake may classify MPI include
# directories as compiler-implicit. CUDA sources are compiled by nvcc, so keep
# the MPI-related implicit directories visible to those targets.
set(GEVD_MPI_INCLUDE_DIRS
  ${MPI_CXX_INCLUDE_DIRS}
  ${MPI_CXX_ADDITIONAL_INCLUDE_DIRS}
  ${MPI_CXX_COMPILER_INCLUDE_DIRS})
foreach(_include_dir IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
  if(EXISTS "${_include_dir}/mpi.h" OR
     EXISTS "${_include_dir}/mpio.h" OR
     EXISTS "${_include_dir}/opal_config.h" OR
     EXISTS "${_include_dir}/mpichconf.h")
    list(APPEND GEVD_MPI_INCLUDE_DIRS "${_include_dir}")
  endif()
endforeach()
list(REMOVE_DUPLICATES GEVD_MPI_INCLUDE_DIRS)

set(CUDA_MATH_ROOT "$ENV{CUDA_MATH_ROOT}" CACHE PATH
    "CUDA math libraries root containing cuSOLVERMp and cuBLASMp")
set(CUDA_COMM_ROOT "$ENV{CUDA_COMM_ROOT}" CACHE PATH
    "CUDA communication libraries root containing NCCL and NVSHMEM")
set(CUSOLVERMP_ROOT "${CUDA_MATH_ROOT}" CACHE PATH
    "cuSOLVERMp installation root")
set(CUBLASMP_ROOT "${CUDA_MATH_ROOT}" CACHE PATH
    "cuBLASMp installation root")
set(NCCL_ROOT "${CUDA_COMM_ROOT}/nccl" CACHE PATH
    "NCCL installation root")
set(NVSHMEM_ROOT "${CUDA_COMM_ROOT}/nvshmem" CACHE PATH
    "NVSHMEM installation root")
set(MKL_ROOT "$ENV{MKLROOT}" CACHE PATH
    "MKL installation root used by the tridiagonal eigensolver")

find_path(MKL_INCLUDE_DIR NAMES mkl_lapacke.h
  HINTS "${MKL_ROOT}" PATH_SUFFIXES include)

find_path(CUSOLVERMP_INCLUDE_DIR NAMES cusolverMp.h
  HINTS "${CUSOLVERMP_ROOT}" "${CUDA_MATH_ROOT}" PATH_SUFFIXES include)
find_library(CUSOLVERMP_LIBRARY NAMES cusolverMp libcusolverMp.so.0
  HINTS "${CUSOLVERMP_ROOT}" "${CUDA_MATH_ROOT}" PATH_SUFFIXES lib lib64)

find_path(CUBLASMP_INCLUDE_DIR NAMES cublasmp.h
  HINTS "${CUBLASMP_ROOT}" "${CUDA_MATH_ROOT}" PATH_SUFFIXES include)
find_library(CUBLASMP_LIBRARY
  NAMES cublasMp cublasmp libcublasMp.so.0 libcublasmp.so.0
  HINTS "${CUBLASMP_ROOT}" "${CUDA_MATH_ROOT}" PATH_SUFFIXES lib lib64)

find_path(NCCL_INCLUDE_DIR NAMES nccl.h
  HINTS "${NCCL_ROOT}" "${CUDA_COMM_ROOT}"
  PATH_SUFFIXES include nccl/include)
find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2
  HINTS "${NCCL_ROOT}" "${CUDA_COMM_ROOT}"
  PATH_SUFFIXES lib lib64 nccl/lib)
find_library(NVSHMEM_HOST_LIBRARY NAMES nvshmem_host libnvshmem_host.so.3
  HINTS "${NVSHMEM_ROOT}" "${CUDA_COMM_ROOT}"
  PATH_SUFFIXES lib lib64 nvshmem/lib)

find_library(MKL_GF_ILP64_LIBRARY NAMES mkl_gf_ilp64 libmkl_gf_ilp64.so.2 libmkl_gf_ilp64.so.3
  HINTS "${MKL_ROOT}" PATH_SUFFIXES lib lib/intel64 lib/intel64_lin)
find_library(MKL_GNU_THREAD_LIBRARY
  NAMES mkl_gnu_thread libmkl_gnu_thread.so.2 libmkl_gnu_thread.so.3
  HINTS "${MKL_ROOT}" PATH_SUFFIXES lib lib/intel64 lib/intel64_lin)
find_library(MKL_CORE_LIBRARY NAMES mkl_core libmkl_core.so.2 libmkl_core.so.3
  HINTS "${MKL_ROOT}" PATH_SUFFIXES lib lib/intel64 lib/intel64_lin)
find_library(GOMP_LIBRARY NAMES gomp libgomp.so.1)

set(_required_dependencies
  CUSOLVERMP_INCLUDE_DIR CUSOLVERMP_LIBRARY
  CUBLASMP_INCLUDE_DIR CUBLASMP_LIBRARY
  NCCL_INCLUDE_DIR NCCL_LIBRARY NVSHMEM_HOST_LIBRARY
  MKL_INCLUDE_DIR MKL_GF_ILP64_LIBRARY MKL_GNU_THREAD_LIBRARY
  MKL_CORE_LIBRARY GOMP_LIBRARY)
foreach(_dependency IN LISTS _required_dependencies)
  if(NOT ${_dependency})
    message(FATAL_ERROR
      "Missing ${_dependency}. Set CUDA_MATH_ROOT, CUDA_COMM_ROOT, and MKL_ROOT.")
  endif()
endforeach()

set(_runtime_libraries
  ${CUSOLVERMP_LIBRARY}
  ${CUBLASMP_LIBRARY}
  ${NCCL_LIBRARY}
  ${NVSHMEM_HOST_LIBRARY}
  ${MKL_GF_ILP64_LIBRARY}
  ${MKL_GNU_THREAD_LIBRARY}
  ${MKL_CORE_LIBRARY})
# Prefer the toolkit target directory over a package-environment top-level
# lib directory.  Conda CUDA packages place the actual CUDA libraries under
# targets/<triplet>/lib while their top-level lib may also contain an unrelated
# MPI implementation.  Recording the narrower directory keeps CUDA's
# transitive dependencies available without changing the MPI selected by
# find_package(MPI).
set(_cuda_runtime_directory "${CUDAToolkit_LIBRARY_DIR}")
foreach(_cuda_library_suffix IN ITEMS lib lib64)
  set(_cuda_runtime_candidate
      "${CUDAToolkit_TARGET_DIR}/${_cuda_library_suffix}")
  if(EXISTS "${_cuda_runtime_candidate}/libcudart.so")
    set(_cuda_runtime_directory "${_cuda_runtime_candidate}")
    break()
  endif()
endforeach()

set(GEVD_RUNTIME_DIRECTORIES "${_cuda_runtime_directory}")
foreach(_library IN LISTS _runtime_libraries)
  get_filename_component(_directory "${_library}" DIRECTORY)
  list(APPEND GEVD_RUNTIME_DIRECTORIES "${_directory}")
endforeach()
list(REMOVE_DUPLICATES GEVD_RUNTIME_DIRECTORIES)
