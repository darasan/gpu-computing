cmake_minimum_required(VERSION 3.12)

# set the project name and version
project(SIFT_CUDA LANGUAGES CXX)

# set CXX standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(inc)

# add all source files
file(GLOB SRC "src/*.cxx" "src/*.cu")

# Linear algebra package
find_package(BLAS)

# Shared memory parallelism
find_package(OpenMP)

# CUDA Libraries
find_package(CUDA)
if(CUDA_FOUND)
  enable_language(CUDA)
  include_directories(${CUDA_INCLUDE_DIRS} )
  add_executable(SIFT_CUDA ${SRC})
endif()
