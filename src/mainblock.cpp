#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <vector>
#include "blockmul.hpp"
#include "gemm.hpp"

template<typename T>
void benchmark_block_vs_gemm() {
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024};
    std::vector<size_t> block_sizes = {8, 16, 32, 64, 128};
    
    // Header
    std::cout << "\n===== Block GEMM vs Regular GEMM Benchmarks (times in milliseconds) =====\n\n";
    std::cout << std::setw(6) << "Size" << " | ";
    std::cout << std::setw(10) << "ijk" << " | ";
    for (const auto& bs : block_sizes) {
        std::cout << std::setw(10) << ("BS=" + std::to_string(bs)) << " | ";
    }
    std::cout << "\n" << std::string(6 + (1 + block_sizes.size()) * 14, '-') << "\n";
    
    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    
    // Run benchmarks for each size
    for (size_t size : sizes) {
        std::cout << std::setw(6) << size << " | ";
        
        size_t M = size, N = size, K = size;
        std::vector<T> A(M * K), B(K * N), C(M * N);
        
        // Initialize with random values
        for (auto& val : A) val = dist(gen);
        for (auto& val : B) val = dist(gen);
        
        // Benchmark regular gemm_ijk
        {
            std::fill(C.begin(), C.end(), T(0));
            auto start = std::chrono::high_resolution_clock::now();
            gemm_ijk(A, B, C, M, N, K);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << std::setw(10) << duration << " | ";
        }
        
        // Benchmark block_gemm_ijk with different block sizes
        for (const auto& bs : block_sizes) {
            std::fill(C.begin(), C.end(), T(0));
            auto start = std::chrono::high_resolution_clock::now();
            block_gemm_ijk(A, B, C, M, N, K, bs);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << std::setw(10) << duration << " | ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Convenience function to run benchmarks for common types
inline void run_block_benchmarks() {
    std::cout << "Single precision (float):" << std::endl;
    benchmark_block_vs_gemm<float>();
    
    std::cout << "Double precision (double):" << std::endl;
    benchmark_block_vs_gemm<double>();
}

int main() {
    run_block_benchmarks();
    return 0;
}