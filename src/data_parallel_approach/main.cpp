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
#include "conj_grad_solve.hpp"

#include "IO.hpp"

using vec = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)


int main(int argc, char **argv)
{
    int n =  324;  // size of the matrix --> later can make this a command line argument,--> I.e. as for this input...
    double error_tol = 1e-4;
    double tolerance = 1e-8; // for cg solver
    std::string matrix_filename = "matrix.txt";
    std::string rhs_filename = "rhs.txt";
    std::string initial_guess_filename = "initial_guess.txt";

    MPI_Init (&argc, &argv);
    int nprocs, rank;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    const mat A = read_matrix(n, matrix_filename);  // here is spot where can use hdf5 in the future for i/o
    const vec b = read_vector(n, rhs_filename);
    const vec initial_guess = read_vector(n, initial_guess_filename);
    int total_iters;

    // NOTE: CONJUGATE GRAD WORKS MUCH NICER WITH SPARSE MATRICES, THAN DENSE ONES!!

    //RECALL THAT I CAN GENERATE LAPALCIAN MATRICES WITH MATLAB, SO COULDD USE THOSE WHEN WANT TO TEST THIS WITH SPARSE MATRICES!

    // NOTE: when testing pick a matrix that acutally is solvable using CG. Use Matlab to test!!!
    // NOTE: currently the matrix sizes must evenly divide into nprocs...
    // note: make sure matrix is big enough for thenumber of processors you are using!

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    vec x = conj_grad_solver(A, b, tolerance, initial_guess, total_iters);  // domain decomposition is done inside the solver

    std::chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = std::chrono::duration_cast<std::chrono::duration<double>>(finish-start);

    if (rank == 0) {

        //std::cout << "Matrix A: " << std::endl;
        //print(A);

        //std::cout << "rhs b: " << std::endl;
        // print(b);

        //std::cout << "solution x: " << std::endl;
        print(x);

        vec A_times_x(x.size());
        //std::cout << "Check A*x = b " << std::endl;
        mat_times_vec(A, x, A_times_x);
        print(A_times_x);

        //------------------------- Verification Test ----------------------------------------------------------------------
        // we will compare the A*x result to the right hand side
        vec error(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            error[i] = abs(A_times_x[i] - b[i]);
        }
        if (*std::max_element(error.begin(), error.end()) > error_tol)
            std::cout << "Error in solution is larger than " << error_tol << std::endl;
        //print(error);

        double cpu_time = time.count();

        std::cout << " CPU time = " << cpu_time << std::endl;

        //------------------------ Write results to HDF5 file------------------------
        write_results_hdf5(x, error, n, cpu_time, tolerance, total_iters);

    }


    MPI_Finalize();
}
