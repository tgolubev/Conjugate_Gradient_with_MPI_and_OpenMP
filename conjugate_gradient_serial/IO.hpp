#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using vec    = std::vector<double>;         // vector
using matrix = std::vector<vec>;            // matrix (=collection of (row) vectors)

void print(const vec &V);

void print(const matrix &A);
