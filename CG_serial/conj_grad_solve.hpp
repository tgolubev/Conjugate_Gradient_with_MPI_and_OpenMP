#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using vec    = std::vector<double>;         // vector
using matrix = std::vector<vec>;            // matrix (=collection of (row) vectors)

vec conj_grad_solver(const matrix &A, const vec &b);
vec mat_times_vec(const std::vector<vec> &A, const vec &v);
vec vec_lin_combo(double a, const vec &u, double b, const vec &v);
double dot_product(const vec &u, const vec &v);
double vector_norm(const vec &v);
