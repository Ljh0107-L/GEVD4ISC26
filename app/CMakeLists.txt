add_executable(gevd_solve gevd_solve.cpp)
target_include_directories(gevd_solve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gevd_solve PRIVATE GEVD4ISC26::gevd4isc26)

add_library(gevd4isc26_app_common STATIC
  common/dense_problem.cpp
  common/distributed_problem.cpp)
target_include_directories(gevd4isc26_app_common PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gevd4isc26_app_common PUBLIC
  GEVD4ISC26::gevd4isc26)

add_executable(gevd_validate gevd_validate.cpp)
target_link_libraries(gevd_validate PRIVATE gevd4isc26_app_common)

add_executable(gevd_benchmark gevd_benchmark.cpp)
target_link_libraries(gevd_benchmark PRIVATE gevd4isc26_app_common)

add_executable(evd_benchmark evd_benchmark.cu)
set_target_properties(evd_benchmark PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(evd_benchmark PRIVATE
  ${PROJECT_SOURCE_DIR}/src/evd/interface
  ${GEVD_MPI_INCLUDE_DIRS})
target_link_libraries(evd_benchmark PRIVATE
  gevd4isc26_standard_evd_engine
  MPI::MPI_CXX
  CUDA::cudart
  CUDA::cublas)

set(_application_targets
  ${GEVD4ISC26_D_AND_C_APP_TARGETS}
  gevd_solve
  gevd_validate
  gevd_benchmark
  evd_benchmark)
set_target_properties(${_application_targets} PROPERTIES
  INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")

set(GEVD4ISC26_APP_TARGETS
  ${_application_targets}
  PARENT_SCOPE)
