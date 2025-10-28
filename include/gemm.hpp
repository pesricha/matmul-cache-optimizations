#ifndef GEMM_HPP
#define GEMM_HPP

#include <vector>
#include <cstddef>

/*
A is M x K
B is K x N
C is M x N
*/

// Transpose matrix B (K x N) to Bt (N x K)
template<typename T>
void transpose(const std::vector<T>& B, std::vector<T>& Bt, std::size_t K, std::size_t N);

// ijk
template<typename T>
void gemm_ijk(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K);

// ikj
template<typename T>
void gemm_ikj(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K);

// jik
template<typename T>
void gemm_jik(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K);

// jki
template<typename T>
void gemm_jki(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K);

// kij
template<typename T>
void gemm_kij(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K);

// kji
template<typename T>
void gemm_kji(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K);

#endif