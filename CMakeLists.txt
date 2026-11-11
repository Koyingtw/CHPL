cmake_minimum_required(VERSION 3.18)

project(CHPL VERSION 1.0.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# NVIDIA H200 / H100 (Hopper)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90a)
endif()
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

# NCCL: set NCCL_ROOT (or NCCL_HOME in the environment) if it is not in a
# default location.
find_path(NCCL_INCLUDE_DIR nccl.h
  HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{NCCL_HOME} ${CUDAToolkit_INCLUDE_DIRS}/..
  PATH_SUFFIXES include)
find_library(NCCL_LIBRARY NAMES nccl
  HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{NCCL_HOME} ${CUDAToolkit_LIBRARY_DIR}/..
  PATH_SUFFIXES lib lib64)
if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL not found; pass -DNCCL_ROOT=<dir>")
endif()
message(STATUS "NCCL: ${NCCL_LIBRARY}")

include(cmake/GEMMul8.cmake)

set(CHPL_CUDA_LIBS
  CHPL::gemmul8 CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cusolver)

# Multi-GPU solvers. The process-grid row count P is fixed at compile time:
#   chpl_p4 : P = 4 (Q = ranks / 4), best for 2+ nodes
#   chpl_p2 : P = 2 (Q = ranks / 2), best for a single 8-GPU node
add_executable(chpl_p4 src/chpl_p4.cu)
add_executable(chpl_p2 src/chpl_p2.cu)
foreach(t chpl_p4 chpl_p2)
  target_include_directories(${t} PRIVATE ${NCCL_INCLUDE_DIR})
  target_link_libraries(${t} PRIVATE ${CHPL_CUDA_LIBS} ${NCCL_LIBRARY} MPI::MPI_CXX)
endforeach()

# Single-GPU solver (no MPI / NCCL)
add_executable(chpl_single src/chpl_single.cu)
target_link_libraries(chpl_single PRIVATE ${CHPL_CUDA_LIBS} CUDA::curand)

foreach(t chpl_p4 chpl_p2 chpl_single)
  target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  set_target_properties(${t} PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
endforeach()

install(TARGETS chpl_p4 chpl_p2 chpl_single RUNTIME DESTINATION bin)
install(PROGRAMS scripts/mpirun_chpl DESTINATION bin)
install(FILES scripts/HPL.dat DESTINATION bin)
