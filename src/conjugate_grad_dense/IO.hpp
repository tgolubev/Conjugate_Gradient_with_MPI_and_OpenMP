#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <mpi.h>
#include <omp.h>

#include <string>
#include "hdf5.h"

using vec    = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)

void print(const vec &V);

void print(const mat &A);
mat read_matrix(const unsigned int n, std::string filename);
vec read_vector(const unsigned int n, std::string filename);
void write_results_hdf5(const char *filename, const vec &solution, const vec &error, const int n, const double cpu_time, const double cpu_time_per_iter, const double tolerance, const int total_iters);
mat read_sub_mat_hdf5(const char *filename, const char *mat_dataset_name, const int n);
mat read_mat_hdf5(const char *filename, const char *mat_dataset_name, const int n);
vec read_vec_hdf5(const char *filename, const char *vec_dataset_name, const int n);
