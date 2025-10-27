# matmul-cache-optimizations
Matrix multiplication cache optimization for single core performance analysis.

## Loop Interchange Optimizations

Loop interchange is a critical optimization technique for matrix multiplication that improves cache locality. By rearranging the order of nested loops, we can achieve significant performance improvements.

For matrix multiplication:

$$
C = A \times B
$$

---

### Performance, Access Patterns, and Miss Ratios

| Order  | Typical Performance (Row-major) | Similar Orders | Total Misses per Iteration | Miss Rate (%) |
|--------|----------------------------------|----------------|-----------------------------|----------------|
| **ikj** | Best / Good                     | kij             | 0.50                        | **16.67%**     |
| **kij** | Best / Good                     | ikj             | 0.50                        | **16.67%**     |
| **ijk** | Medium / Decent                 | jik             | 1.25                        | **62.50%**     |
| **jik** | Medium / Decent                 | ijk             | 1.25                        | **62.50%**     |
| **kji** | Poor / Slow                     | jki             | 2.00                        | **66.67%**     |
| **jki** | Poor / Slow                     | kji             | 2.00                        | **66.67%**     |

---

### Detailed Access and Miss Calculations

#### ikj / kij variants
- Loads per iteration: 2  
- Stores per iteration: 1  
- Total accesses: 3  
- Total misses: 0.50  
- Miss % per access: 0.50 / 3 = 16.67%  
- Per-array miss details:  
  - A: 0.00 / 1 → 0.00%  
  - B: 0.25 / 1 → 25.00%  
  - C: 0.25 / 2 → 12.50%  

Interpretation:  
Good reuse of A and C, moderate reuse of B.  
Best overall spatial and temporal locality.

---

#### ijk / jik variants
- Loads per iteration: 2  
- Stores per iteration: 0  
- Total accesses: 2  
- Total misses: 1.25  
- Miss % per access: 1.25 / 2 = 62.50%  
- Per-array miss details:  
  - A: 0.25 / 1 → 25.00%  
  - B: 1.00 / 1 → 100.00%  
  - C: not accessed  

Interpretation:  
Matrix B accessed column-wise in row-major memory layout, causing frequent cache misses.

---

#### jki / kji variants
- Loads per iteration: 2  
- Stores per iteration: 1  
- Total accesses: 3  
- Total misses: 2.00  
- Miss % per access: 2.00 / 3 = 66.67%  
- Per-array miss details:  
  - A: 1.00 / 1 → 100.00%  
  - B: not accessed  
  - C: 1.00 / 2 → 50.00%  

Interpretation:  
High stride access in A and C leads to very poor cache performance and the highest miss rate.

---

### Summary

- ikj, kij: Lowest miss rate (~16.7%), best locality and cache reuse.  
- ijk, jik: Moderate miss rate (~62.5%), poor reuse of B due to column-wise access.  
- jki, kji: Highest miss rate (~66.7%), poor reuse of A and C due to strided access.

---

## Building the Project

To build and run the project:

```bash
# Clone the repository
git clone https://github.com/pesricha/matmul-cache-optimizations
cd matmul-cache-optimizations

# Create build directory
mkdir build && cd build

# Configure and build with CMake
cmake ..
make

# Run the program
./matmul-cache-optimizations
```

### Requirements
- CMake 3.10 or higher  
- C++17 compatible compiler (GCC, Clang, or MSVC)
