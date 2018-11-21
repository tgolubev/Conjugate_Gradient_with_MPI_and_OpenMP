#ifndef INCOMPLETE_CHOLESKY_H
#define INCOMPLETE_CHOLESKY_H

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <mpi.h>

using vec    = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)

void incomplete_cholesky(mat &A);




#endif // INCOMPLETE_CHOLESKY_H
