#include <blockmul.hpp>
#include <cstddef>

template<typename T>
void block_gemm_ijk(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C,
              std::size_t M, std::size_t N, std::size_t K, std::size_t BS) 
{
    for (size_t ii = 0 ; ii < M ; ii+=BS) {
        for (size_t jj = 0 ; jj < N ; jj+=BS) {
            for (size_t kk = 0 ; kk < K ; kk+=BS) {
                
                for (size_t i = ii; i < ii + BS && i < M; i++) {
                    for (size_t j =  jj; j < jj + BS && j < N; j++) {
                        
                        T c_ij = C[i*N + j];

                        for (size_t k =  kk; k < kk + BS && k < K; k++) {
                            c_ij += A[i*K + k] * B[k*N + j];
                        }
                        C[i*N + j] = c_ij;
                    }
                }

            }
        }
    }
}


// Explicit Instantiation
template void block_gemm_ijk<float>(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, std::size_t M, std::size_t N, std::size_t K, std::size_t BS);
template void block_gemm_ijk<double>(const std::vector<double> &A, const std::vector<double> &B, std::vector<double> &C, std::size_t M, std::size_t N, std::size_t K, std::size_t BS);