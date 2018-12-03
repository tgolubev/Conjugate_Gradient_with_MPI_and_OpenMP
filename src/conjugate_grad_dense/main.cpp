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
    int n =  2824;  // size of the matrix --> later can make this a command line argument,--> I.e. as for this input...
    double error_tol = 1e-4;
    double tolerance = 1e-8; // for cg solver

    int num_solves = 3;  // number of solves of CG to do, for better statistics of cpu time

//    std::string matrix_filename = "matrix.txt";
//    std::string rhs_filename = "rhs.txt";
//    std::string initial_guess_filename = "initial_guess.txt";

    // names inside hd5 files need to be of char format
    const char *h5_filename = "cg.h5";
    const char *mat_dataset_name = "matrix";
    const char *rhs_dataset_name = "rhs";
    const char *guess_dataset_name = "initial_guess";

    MPI_Init (&argc, &argv);
    int nprocs, rank;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    const mat sub_A = read_sub_mat_hdf5(h5_filename, mat_dataset_name, n);  // here is spot where can use hdf5 in the future for i/o
    const mat A = read_mat_hdf5(h5_filename, mat_dataset_name, n);
    const vec b = read_vec_hdf5(h5_filename, rhs_dataset_name, n);
    const vec initial_guess = read_vec_hdf5(h5_filename, guess_dataset_name, n);
    int total_iters;
    vec x;


    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_solves; i++) {

        x = conj_grad_solver(sub_A, b, tolerance, initial_guess, total_iters);  // domain decomposition is done inside the solver
        //x = conj_grad_solver_omp_sections(sub_A, b, tolerance, initial_guess, total_iters);
        //x = conj_grad_solver_omp_tasks(sub_A, b, tolerance, initial_guess, total_iters);

    }

    std::chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = std::chrono::duration_cast<std::chrono::duration<double>>(finish-start);

    if (rank == 0) {

        //std::cout << "Matrix A: " << std::endl;
        //print(A);

        //std::cout << "rhs b: " << std::endl;
        // print(b);

        //std::cout << "solution x: " << std::endl;
        //print(x);

        vec A_times_x(x.size());
        //std::cout << "Check A*x = b " << std::endl;
        mat_times_vec(A, x, A_times_x);
        //print(A_times_x);

        //------------------------- Verification Test ----------------------------------------------------------------------
        // we will compare the A*x result to the right hand side
        vec error(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            error[i] = abs(A_times_x[i] - b[i]);
        }
        if (*std::max_element(error.begin(), error.end()) > error_tol)
            std::cout << "Error in solution is larger than " << error_tol << std::endl;
        //print(error);

        double cpu_time = time.count()/num_solves; // is cpu time per CG solve
        double cpu_time_per_iter = cpu_time/total_iters;

        std::cout << " Total CPU time = " << cpu_time*num_solves << std::endl;
        std::cout << " CPU time per CG solve = " << cpu_time << std::endl;
        std::cout << " CPU time per iter = " << cpu_time_per_iter << std::endl;

        //------------------------ Write results to HDF5 file------------------------
        write_results_hdf5(h5_filename, x, error, n, cpu_time, cpu_time_per_iter, tolerance, total_iters);

    }

    MPI_Finalize();
}
