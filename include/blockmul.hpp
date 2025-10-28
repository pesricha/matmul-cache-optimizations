#ifndef BLOCKMUL_HPP
#define BLOCKMUL_HPP

#include <vector>
#include <cstddef>

template<typename T>
void block_gemm_ijk(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K, std::size_t BS);

#endif