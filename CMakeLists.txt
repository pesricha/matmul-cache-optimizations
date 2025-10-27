cmake_minimum_required(VERSION 3.10)

# Project name
project(matmul-cache-optimizations)

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
add_executable(matmul-cache-optimizations ${SRC_FILES})

# Add compiler warnings and optimization flags
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(matmul-cache-optimizations PRIVATE 
        -Wall -Wextra -pedantic -O3 -march=native -funroll-loops
    )
elseif (MSVC)
    target_compile_options(matmul-cache-optimizations PRIVATE 
        /W4 /O2
    )
endif()

# Optionally: installation
# install(TARGETS matmul-cache-optimizations DESTINATION bin)
# install(DIRECTORY include/ DESTINATION include)
