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
    int n =  500;  // size of the matrix --> later can make this a command line argument,--> I.e. as for this input...
    std::string matrix_filename = "matrix.txt";
    std::string rhs_filename = "rhs.txt";

    // test openmp by printing out number of threads we have
    /*
    # pragma omp parallel
    {
       printf("Num OpenMP threads: %d\n", omp_get_num_threads());
    }
    */

   MPI_Init (&argc, &argv);
   int nprocs, rank;
   MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank (MPI_COMM_WORLD, &rank);

   mat A = read_matrix(n, matrix_filename);  // here is spot where can use hdf5 in the future for i/o
   vec b = read_vector(n, rhs_filename);

   // NOTE: when testing pick a matrix that acutally is solvable using CG. Use Matlab to test!!!
   // NOTE: currently the matrix sizes must evenly divide into nprocs...
   // note: make sure matrix is big enough for thenumber of processors you are using!

  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

   vec x = conj_grad_solver(A, b);  // domain decomposition is done inside the solver

   std::chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
   std::chrono::duration<double> time = std::chrono::duration_cast<std::chrono::duration<double>>(finish-start);


   if (rank == 0) {
   
       //std::cout << "Matrix A: " << std::endl;
       //print(A);
        
       //std::cout << "rhs b: " << std::endl;
       // print(b);
        
        //std::cout << "solution x: " << std::endl;
        //print(x);
        
        // WE CAN USE AN AUTOMATIC CHECK HERE LATER, TO COMPARE THE EQUALITY OF THE RESULTS, i.e. check if are < a tolerance,
        // otherwise output an error.

        //std::cout << "Check A*x = b " << std::endl;
        //print(mat_times_vec(A, x));

        std::cout << " CPU time = " << time.count() << std::endl;
    
   }

   
   MPI_Finalize();
}





