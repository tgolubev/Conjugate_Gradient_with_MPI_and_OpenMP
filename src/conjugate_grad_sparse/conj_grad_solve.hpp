/*****************************************************************************
 Definitions of the main functions for the parallel Conjugate Gradient solver
 calculations.

 Author: Timofey Golubev

*******************************************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <mpi.h>
#include <parallel/numeric> # for using paralle versions of inner_product

using vec    = std::vector<double>;      // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)

//! The fastest implementation of conjugate gradient algorithm, using data-based parallelism only
vec conj_grad_solver(const mat &sub_A_values, const mat &sub_A_indices, const vec &b, const double tolerance, const vec &initial_guess, int &total_iters);

//! CG solver with added task-based parallelism through tasks attempt. (Is a little slower than the conj_grad_solver).
vec conj_grad_solver_omp_tasks(const mat &sub_A_values, const mat &sub_A_indices, const vec &b, const double tolerance, const vec &initial_guess, int &total_iters);

//! CG solver with added task-based parallelism thorugh sections attempt. (Is a little slower than the conj_grad_solver).
vec conj_grad_solver_omp_sections(const mat &sub_A, const vec &b, const double tolerance, const vec &initial_guess, int &total_iters);

//! Matrix times vector product
void mat_times_vec(const mat &sub_A_values, const mat &sub_A_indices, const vec &v, vec &result);

//! Linear combination of vectors: a*u + b*v
void vec_lin_combo(double a, const vec &u, double b, const vec &v, vec &result);

//! Dot product of full vectors
double dot_product(const vec &u, const vec &v);

//! Norm of a full vector sqrt(v*v)
double vector_norm(const vec &v);

//! Dot product of vectors which are divided among the MPI ranks
double mpi_dot_product(const vec &sub_u, const vec &sub_v, double product);

//! Norm of a vector which is divided among the MPI ranks
double mpi_vector_norm(const vec &sub_v, double norm_r);
