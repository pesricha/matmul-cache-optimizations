
#include <chrono>
#include <iostream>
#include <iomanip>
#include <functional>
#include <string>
#include <random>
#include <map>
#include "gemm.hpp"

template<typename T>
void benchmark_gemm() {
    std::vector<size_t> sizes = {64, 128, 256, 512, 1024};
    
    // Create a map of functions to benchmark
    std::map<std::string, std::function<void(const std::vector<T>&, const std::vector<T>&, 
                                            std::vector<T>&, std::size_t, std::size_t, std::size_t)>> functions = {
        {"ijk", gemm_ijk<T>},
        {"ikj", gemm_ikj<T>},
        {"jik", gemm_jik<T>},
        {"jki", gemm_jki<T>},
        {"kij", gemm_kij<T>},
        {"kji", gemm_kji<T>}
    };
    
    // Header
    std::cout << "\n===== GEMM Benchmarks (times in milliseconds) =====\n\n";
    std::cout << std::setw(6) << "Size" << " | ";
    for (const auto& f : functions) {
        std::cout << std::setw(10) << f.first << " | ";
    }
    std::cout << "\n" << std::string(6 + functions.size() * 14, '-') << "\n";
    
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
        
        // Benchmark each function
        for (const auto& f : functions) {
            // Reset C
            std::fill(C.begin(), C.end(), T(0));
            
            // Time the function
            auto start = std::chrono::high_resolution_clock::now();
            f.second(A, B, C, M, N, K);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << std::setw(10) << duration << " | ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Convenience function to run benchmarks for common types
inline void run_gemm_benchmarks() {
    std::cout << "Single precision (float):" << std::endl;
    benchmark_gemm<float>();
    
    std::cout << "Double precision (double):" << std::endl;
    benchmark_gemm<double>();
}

int main() {
    run_gemm_benchmarks();
    return 0;
}