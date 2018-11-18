#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>

using vec    = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)

void print(const vec &V);

void print(const mat &A);
mat read_matrix(const unsigned int n, std::string filename);
vec read_vector(const unsigned int n, std::string filename);
