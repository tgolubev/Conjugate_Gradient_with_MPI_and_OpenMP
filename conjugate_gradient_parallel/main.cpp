#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <mpi.h>

#include "IO.hpp"
#include "conj_grad_solve.hpp"


using vec = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)


int main()
{
    
   MPI_Init (&nargs, &args);
   MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank (MPI_COMM_WORLD, &rank);
   
   mat A = { { 4, 1 }, { 1, 3 } };
   vec b = { 1, 2 };
   
   // Domain decomposition will be done here, so we will only pass the sub-matrices to the conj_grad_solver.
   // However we will pass the entire b vector (at least for now), b/c it is needed for the multiplication...

   vec x = conj_grad_solver( A, b );
   
   if (rank == 0) {
   
        std::cout << "Matrix A: " << std::endl;
        print(A);
        
        std::cout << "rhs b: " << std::endl;
        print(b);
        
        std::cout << "solution x: " << std::endl;
        print(x);
        
        std::cout << "Check A*x = b " << std::endl;
        print(mat_times_vec(A, x));
    
   }
   
   MPI_Finalize();
}





