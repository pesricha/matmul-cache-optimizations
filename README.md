# gemm-cache-optimizations
GEMM cache optimization for speedup without using multiple CPUs/GPUs.

## Loop Interchange Optimizations

Loop interchange is a critical optimization technique for matrix multiplication that improves cache locality. By rearranging the order of nested loops, we can achieve significant performance improvements.

### Different Loop Orders

For matrix multiplication:

$C = A \times B$

The typical six loop orders are:

1. `ijk`
2. `ikj`
3. `jik`
4. `jki`
5. `kij`
6. `kji`

#### Performance and Access Patterns

| Order  | Typical Performance (Row-major)             | Similar Orders      |
|--------|---------------------------------------------|---------------------|
| **ikj**  | Best / Good                                 | kij                 |
| ijk    | Medium / Decent                              | jik                 |
| jik    | Medium / Decent                              | ijk                 |
| kij    | Best / Good                                 | ikj                 |
| kji    | Poor / Slow                                  | jki                 |
| jki    | Poor / Slow                                  | kji                 |

✅ **`ikj`** and **`kij`** are typically the fastest for row-major memory layout because they:
- Access matrix `A` row-wise (good cache reuse).
- Access matrix `B` row-wise (good for row-major storage).
- Update matrix `C` contiguously.

✅ **`ijk`** and **`jik`** are decent but suffer from poor access patterns for matrix `B`.

✅ **`jki`** and **`kji`** typically have the worst cache usage and performance in row-major layouts.

---
## Building the Project

To build and run the project:

```bash
# Clone the repository
git clone https://github.com/pesricha/gemm-cache-optimizations
cd gemm-cache-optimizations

# Create build directory
mkdir build && cd build

# Configure and build with CMake
cmake ..
make

# Run the program
./gemm-cache-optimizations
```

### Requirements

- CMake 3.10 or higher
- C++17 compatible compiler (GCC, Clang, or MSVC)
