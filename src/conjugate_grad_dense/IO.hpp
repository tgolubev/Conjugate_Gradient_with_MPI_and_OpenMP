/*****************************************************************************
 Definitions of the input/output functions for the conjugate gradient solver.
 Here are functions for HDF5, printing to text file (not currently used
 by the solver), and printing data to terminal in a readable way.

 Author: Timofey Golubev

*******************************************************************************/

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

using vec    = std::vector<double>;      // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)

//! Print a vector to terminal
void print(const vec &V);

//! Print a matrix to terminal
void print(const mat &A);

//! Read a matrix from a text file
mat read_matrix(const unsigned int n, std::string filename);

//! Read a vector from a text file
vec read_vector(const unsigned int n, std::string filename);

//! Read sub-matrix (horizontal slice with size depending on # of MPI ranks) from HDF5 file
mat read_sub_mat_hdf5(const char *filename, const char *mat_dataset_name, const int n);

//! Read full matrix from HDF5 file
mat read_mat_hdf5(const char *filename, const char *mat_dataset_name, const int n);

//! Read vector from HDF5 file
vec read_vec_hdf5(const char *filename, const char *vec_dataset_name, const int n);

//! Write CG results to HDF5 file
void write_results_hdf5(const char *filename, const vec &solution, const vec &error, const int n, const double cpu_time, const double cpu_time_per_iter, const double tolerance, const int total_iters);

