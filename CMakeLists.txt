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
    src/gemm.cpp
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


# Source files for the second executable
set(BLOCK_SRC_FILES
    src/mainblock.cpp
    src/blockmul.cpp
    src/gemm.cpp
)

# Create second executable
add_executable(matmul-block-optimizations ${BLOCK_SRC_FILES})

# Add compiler warnings and optimization flags for the second executable
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(matmul-block-optimizations PRIVATE 
        -Wall -Wextra -pedantic -O3 -march=native -funroll-loops
    )
elseif (MSVC)
    target_compile_options(matmul-block-optimizations PRIVATE 
        /W4 /O2
    )
endif()

# Build both executables by default
add_custom_target(build_all ALL DEPENDS matmul-cache-optimizations matmul-block-optimizations)

# Optionally: installation
# install(TARGETS matmul-cache-optimizations DESTINATION bin)
# install(DIRECTORY include/ DESTINATION include)
