#pragma once

#include <vector>
#include <cstddef>

/*
A is M x K
B is K x N
C is M x N
*/

// Transpose matrix B (K x N) to Bt (N x K)
template<typename T>
void transpose(const std::vector<T>& B, std::vector<T>& Bt, std::size_t K, std::size_t N)
{
    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t n = 0; n < N; ++n) {
            Bt[n * K + k] = B[k * N + n];
        }
    }
}

// ijk
template<typename T>
void gemm_ijk(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K)
{
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            for (std::size_t k = 0; k < K; ++k)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
}

// ikj
template<typename T>
void gemm_ikj(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K)
{
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t j = 0; j < N; ++j)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

// jik
template<typename T>
void gemm_jik(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K)
{
    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t k = 0; k < K; ++k)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
}

// jki
template<typename T>
void gemm_jki(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K)
{
    for (std::size_t j = 0; j < N; ++j)
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t i = 0; i < M; ++i)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

// kij
template<typename T>
void gemm_kij(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K)
{
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t i = 0; i < M; ++i)
            for (std::size_t j = 0; j < N; ++j)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

// kji
template<typename T>
void gemm_kji(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K)
{
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j)
            for (std::size_t i = 0; i < M; ++i)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

