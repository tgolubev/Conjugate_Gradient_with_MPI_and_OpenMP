#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <mpi.h>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include <stddef.h>
#include <omp.h>

#include "IO.hpp"
#include "conj_grad_solve.hpp"


using vec = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)


int main(int argc, char **argv)
{   
    int n =  16;  // size of the matrix --> later can make this a command line argument,--> I.e. as for this input...

    // test openmp by printing out number of threads we have
    # pragma omp parallel
    {
       printf("Num OpenMP threads: %d\n", omp_get_num_threads());
    }

   MPI_Init (&argc, &argv);
   int nprocs, rank;
   MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank (MPI_COMM_WORLD, &rank);

   mat A = read_matrix(n);  // here is spot where can use hdf5 in the future for i/o
   vec b = { -14, 1, 15, 28, 40, 51, 61, 70, 78, 85, 91, 96, 100, 103, 105, 106};

   /* solution is:
    * ans =

          Columns 1 through 7

          -0.499999439030162   0.249996074971279   0.625010978626115   0.812487439164702   0.906249956854653   0.953134713097334   0.976563619939453

          Columns 8 through 14

           0.988273307253737   0.994135639693085   0.997074063488401   0.998541650818801   0.999267843179264   0.999625529647945   0.999804892558474

          Columns 15 through 16

           0.999896603369773   0.999937123101355

                   */


   // NOTE: when testing pick a matrix that acutally is solvable using CG. Use Matlab to test!!!

   // NOTE: currently the matrix sizes must evenly divide into nprocs...

   // note: make sure matrix is big enough for thenumber of processors you are using!

   
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

   vec x = conj_grad_solver(A, b);  // domain decomposition is done inside the solver

   std::chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> time = std::chrono::duration_cast<std::chrono::duration<double>>(finish-start);
   std::cout << " CPU time = " << time.count() << std::endl;
   

   if (rank == 0) {
   
        std::cout << "Matrix A: " << std::endl;
        print(A);
        
       //std::cout << "rhs b: " << std::endl;
       // print(b);
        
        std::cout << "solution x: " << std::endl;
        print(x);
        
        std::cout << "Check A*x = b " << std::endl;
        print(mat_times_vec(A, x));
    
   }

   
   MPI_Finalize();
}





