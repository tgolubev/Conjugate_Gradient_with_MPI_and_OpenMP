# Parallel Conjugate Gradient algorithm with MPI plus OpenMP

An parallel implementation of the conjugate gradient algorithm using a hybrid of distributed (MPI) and shared (OpenMP) memory approach for both sparse and dense matrices.
For sparse matrices, compressed row storage (CRS) is used.


## MPI and OpenMP features used:
* MPI_Allreduce for dot products
* MPI_Allgatherv for matrix-vector product
* MPI_Gatherv to get solution vector
* omp parallel for and/or omp simd in matrix-vector product
* omp parallel for reduction for sub-vector dot products

Files in Hierarchical Data Format (HDF5) are used for parallel I/O and each node reads only the portion of the matrix which it needs.

## Features attempted but which did not result in any performance improvements:
* omp tasks or sections to simultaneously calculate lines 15 and 16 in CG algorithm.
 -  Same CPU times
* omp simd vectorization instead of the OpenMP parallel for
     - Slower CPU times, same as when no simd nor parallel for
* omp parallel for simd: vectorization combined with multithreading
     -  Same CPU times


