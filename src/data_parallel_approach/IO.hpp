#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include <string>
#include "hdf5.h"

using vec    = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)

void print(const vec &V);

void print(const mat &A);
mat read_matrix(const unsigned int n, std::string filename);
vec read_vector(const unsigned int n, std::string filename);
void write_results_hdf5(const vec &solution, const vec &error, const int n, const double cpu_time, const double tolerance, const int total_iters);
void write_num_iters_hdf5(const int iters);
