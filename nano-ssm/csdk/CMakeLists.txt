cmake_minimum_required(VERSION 3.10)
project(ncssm_sdk C)

set(CMAKE_C_STANDARD 99)

# Optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3")

# Include dirs
include_directories(include)

# Library
add_library(ncssm STATIC
    src/ncssm.c
    src/ssm_scan.c
    src/features.c
    src/linear.c
    src/activations.c
    src/utils.c
)
target_link_libraries(ncssm m)

# Test executable
add_executable(test_ncssm test/test_ncssm.c)
target_link_libraries(test_ncssm ncssm m)
