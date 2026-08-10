cmake_minimum_required(VERSION 3.24)

project(GEVD4ISC26 VERSION 0.1.0 LANGUAGES CXX CUDA)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are disabled. Use: cmake -S . -B build")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING
    "CUDA architectures to compile")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

include(src/cmake/Dependencies.cmake)

add_subdirectory(src)
add_subdirectory(app)

install(TARGETS
  gevd4isc26
  ${GEVD4ISC26_INTERNAL_TARGETS}
  EXPORT GEVD4ISC26Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${GEVD4ISC26_APP_TARGETS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(_package_directory "${CMAKE_INSTALL_LIBDIR}/cmake/GEVD4ISC26")
configure_package_config_file(
  src/cmake/GEVD4ISC26Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/GEVD4ISC26Config.cmake"
  INSTALL_DESTINATION "${_package_directory}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/GEVD4ISC26ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(EXPORT GEVD4ISC26Targets
  NAMESPACE GEVD4ISC26::
  DESTINATION "${_package_directory}")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/GEVD4ISC26Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/GEVD4ISC26ConfigVersion.cmake"
  DESTINATION "${_package_directory}")
