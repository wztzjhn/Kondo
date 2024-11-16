cmake_minimum_required(VERSION 3.18)

project(kondo)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# -- COMPILE OPTIONS -------------------------------------------
# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# Performance/debug options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -gdwarf-3")
add_definitions(-D_GLIBCXX_DEBUG=1) # libstdc++
add_definitions(-D_LIBCPP_DEBUG2=1) # libc++


# -- DEPENDENCIES -------------------------------------------

# MPI
find_package(MPI)
if (WITH_MPI)
  include_directories(${MPI_INCLUDE_PATH})
  set(LIBS ${LIBS} ${MPI_LIBRARIES})
  set(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_LINK_FLAGS}")
  add_definitions(-DWITH_MPI)
endif()

# CUDA
find_package(CUDA 11.3)
if (CUDA_FOUND)
  message(STATUS "Found CUDA " ${CUDA_VERSION})
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  set(LIBS ${LIBS} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})
  add_definitions(-DWITH_CUDA)
  if (APPLE)
    # With recent versions of Xcode, must explicitly use clang
    set(CUDA_HOST_COMPILER /usr/bin/clang)
  endif()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2;-Xcompiler=-fPIC;
    -gencode=arch=compute_52,code=sm_52;
    -gencode=arch=compute_60,code=sm_60;
    -gencode=arch=compute_61,code=sm_61;
    -gencode=arch=compute_70,code=sm_70;
    -gencode=arch=compute_70,code=compute_70)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

# Armadillo
find_package(Armadillo REQUIRED)
include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})
set(LIBS ${LIBS} ${ARMADILLO_LIBRARIES})

# FastKPM
find_package(fastkpm REQUIRED)
include_directories(SYSTEM ${FASTKPM_INCLUDE_DIRS})
set(LIBS ${LIBS} ${FASTKPM_LIBRARIES})

#cpptoml
find_package(CPPTOML REQUIRED)
include_directories(SYSTEM ${CPPTOML_INCLUDE_DIRS})

# Boost
find_package(Boost 1.55.0 REQUIRED COMPONENTS filesystem system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})


# -- BUILD -------------------------------------------

include_directories(src)
add_subdirectory(src)


# -- REPORT -------------------------------------------

message("")
message("Configuration results")
message("---------------------")
message("C++ compiler        : ${CMAKE_CXX_COMPILER}")
message("C++ compiler flags  : ${CMAKE_CXX_FLAGS}")

get_property(INC_DIRS DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("LIBS:     ${LIBS}")
message("INCLUDES: ${INC_DIRS}")
message("WITH_MPI: ${MPI_FOUND}")
message("")
