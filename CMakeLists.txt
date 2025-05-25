cmake_minimum_required(VERSION 3.10)

# Project name
project(gemm-cache-optimizations)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

# Source files
set(SRC_FILES
    src/main.cpp
)

# Create executable
add_executable(gemm-cache-optimizations ${SRC_FILES})

# Add compiler warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gemm-cache-optimizations PRIVATE -Wall -Wextra -pedantic)
elseif (MSVC)
    target_compile_options(gemm-cache-optimizations PRIVATE /W4)
endif()

# Optionally: installation
# install(TARGETS gemm-cache-optimizations DESTINATION bin)
# install(DIRECTORY include/ DESTINATION include)
