cmake_minimum_required(VERSION 3.24)

set(_d_and_c_standalone OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(GEVD4ISC26_D_AND_C VERSION 0.1.0 LANGUAGES CXX)
  set(_d_and_c_standalone ON)
  include(GNUInstallDirs)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

  find_package(Threads REQUIRED)
  set(MKL_ROOT "$ENV{MKLROOT}" CACHE PATH
      "Intel MKL root used by the CPU D&C eigensolver")
  find_path(MKL_INCLUDE_DIR NAMES mkl_lapacke.h
    HINTS "${MKL_ROOT}" PATH_SUFFIXES include)
  find_library(MKL_GF_ILP64_LIBRARY NAMES mkl_gf_ilp64 libmkl_gf_ilp64.so.2
    HINTS "${MKL_ROOT}" PATH_SUFFIXES lib lib/intel64 lib/intel64_lin)
  find_library(MKL_GNU_THREAD_LIBRARY
    NAMES mkl_gnu_thread libmkl_gnu_thread.so.2
    HINTS "${MKL_ROOT}" PATH_SUFFIXES lib lib/intel64 lib/intel64_lin)
  find_library(MKL_CORE_LIBRARY NAMES mkl_core libmkl_core.so.2
    HINTS "${MKL_ROOT}" PATH_SUFFIXES lib lib/intel64 lib/intel64_lin)
  find_library(GOMP_LIBRARY NAMES gomp libgomp.so.1)
  foreach(_dependency IN ITEMS
      MKL_INCLUDE_DIR
      MKL_GF_ILP64_LIBRARY
      MKL_GNU_THREAD_LIBRARY
      MKL_CORE_LIBRARY
      GOMP_LIBRARY)
    if(NOT ${_dependency})
      message(FATAL_ERROR "Missing ${_dependency}; set MKL_ROOT or MKLROOT.")
    endif()
  endforeach()
endif()

add_library(gevd4isc26_d_and_c SHARED
  d_and_c_solver.cpp)
add_library(GEVD4ISC26::d_and_c ALIAS gevd4isc26_d_and_c)
set_target_properties(gevd4isc26_d_and_c PROPERTIES
  EXPORT_NAME d_and_c
  POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(gevd4isc26_d_and_c PRIVATE MKL_ILP64)
target_compile_options(gevd4isc26_d_and_c PRIVATE
  -Wall -Wextra -Wpedantic)
target_include_directories(gevd4isc26_d_and_c
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/gevd4isc26>
  PRIVATE
    ${MKL_INCLUDE_DIR})
target_link_libraries(gevd4isc26_d_and_c PRIVATE
  ${MKL_GF_ILP64_LIBRARY}
  ${MKL_GNU_THREAD_LIBRARY}
  ${MKL_CORE_LIBRARY}
  ${GOMP_LIBRARY}
  Threads::Threads
  ${CMAKE_DL_LIBS}
  m)

set(_d_and_c_runtime_directories)
foreach(_library IN ITEMS
    "${MKL_GF_ILP64_LIBRARY}"
    "${MKL_GNU_THREAD_LIBRARY}"
    "${MKL_CORE_LIBRARY}")
  get_filename_component(_directory "${_library}" DIRECTORY)
  list(APPEND _d_and_c_runtime_directories "${_directory}")
endforeach()
list(REMOVE_DUPLICATES _d_and_c_runtime_directories)
set_target_properties(gevd4isc26_d_and_c PROPERTIES
  BUILD_RPATH "${_d_and_c_runtime_directories}"
  INSTALL_RPATH "$ORIGIN;${_d_and_c_runtime_directories}")

add_executable(d_and_c_benchmark d_and_c_benchmark.cpp)
target_compile_options(d_and_c_benchmark PRIVATE
  -Wall -Wextra -Wpedantic)
target_link_libraries(d_and_c_benchmark PRIVATE
  GEVD4ISC26::d_and_c)
set_target_properties(d_and_c_benchmark PROPERTIES
  INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

install(FILES d_and_c_solver.hpp
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gevd4isc26")
if(_d_and_c_standalone)
  install(TARGETS gevd4isc26_d_and_c d_and_c_benchmark
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
else()
  set(GEVD4ISC26_D_AND_C_TARGETS
    gevd4isc26_d_and_c
    PARENT_SCOPE)
  set(GEVD4ISC26_D_AND_C_APP_TARGETS
    d_and_c_benchmark
    PARENT_SCOPE)
endif()
