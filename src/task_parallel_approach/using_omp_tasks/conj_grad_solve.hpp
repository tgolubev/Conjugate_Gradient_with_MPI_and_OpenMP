#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <mpi.h>
#include <parallel/numeric> # for using paralle versions of inner_product

using vec    = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)

vec conj_grad_solver(const mat &A, const vec &b, const vec &initial_guess);
void mat_times_vec(const std::vector<vec> &A, const vec &v, vec &result);
void vec_lin_combo(double a, const vec &u, double b, const vec &v, vec &result);
double dot_product(const vec &u, const vec &v);
double vector_norm(const vec &v);
double mpi_dot_product(const vec &sub_u, const vec &sub_v, double product);
double mpi_vector_norm(const vec &sub_v, double norm_r);