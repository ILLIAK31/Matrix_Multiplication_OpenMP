# Matrix_Multiplication_OpenMP
This C++ project demonstrates optimized matrix multiplication techniques using OpenMP for parallel processing, aiming to improve performance on multi-core systems. It compares several approaches for matrix multiplication, measuring their efficiency and highlighting optimizations like matrix transposition and tiling. This project serves as an excellent starting point for developers looking to understand and implement high-performance matrix operations.
# Key Features
Multi-Method Parallel Matrix Multiplication : The program supports various multiplication strategies, enabling users to evaluate the impact of each method on performance Standard Parallel Multiplication Divides the task among threads using a straightforward approach. Parallel Multiplication with Transposition : Reduces cache misses and improves memory access patterns by transposing one of the matrices Tiled Parallel Multiplication Divides matrices into smaller tiles, leveraging data locality for enhanced cache performance. 
User Configurable Parameters : Users can specify the matrix size, the number of threads, and the tile size, offering flexibility to test performance across different hardware setups and problem sizes. Built-in Performance Benchmarking Each multiplication method logs the time taken for completion, providing insights into the efficiency gains of each optimization.

#### To compile this program in the Linux terminal, use the following command:
`g++ -std=c++11 -pthread -O3 -o App App.cpp`
<ul>
   <li>-std=c++11 : Specifies the use of the C++11 standard.</li>
   <li>-pthread : Enables multi-threading, allowing the program to run more efficiently when working with large matrices.</li>
   <li>-O3 : This is the highest level of optimization in GCC, which enables all optimizations that do not involve a space-speed tradeoff. It aggressively optimizes the code to improve performance, making it 
    especially useful for computationally intensive tasks like multiplying large matrices.</li>
</ul>

### Optimization Techniques
Matrix Transposition Matrix b is transposed before multiplication, allowing for more sequential memory access This reduces cache misses, particularly effective for larger matrices Tiling Tiling divides matrices into smaller submatrices (tiles) By operating on tiles, each thread can work on smaller, cache-friendly chunks of data, minimizing memory latency and improving cache performance
OpenMP Parallelism OpenMP directives (#pragma omp) are used extensively to parallelize loops across multiple threads OpenMP is ideal for shared-memory parallelism, making it suitable for multi-core CPUs
