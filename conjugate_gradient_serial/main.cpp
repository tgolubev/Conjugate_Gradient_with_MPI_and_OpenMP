#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "IO.hpp"
#include "conj_grad_solve.hpp"


using vec = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)


int main()
{
   mat A = { { 4, 1 }, { 1, 3 } };
   vec b = { 1, 2 };

   vec x = conj_grad_solver( A, b );
   
   std::cout << "Matrix A: " << std::endl;
   print(A);
   
   std::cout << "rhs b: " << std::endl;
   print(b);
   
   std::cout << "solution x: " << std::endl;
   print(x);
   
   std::cout << "Check A*x = b " << std::endl;
   print(mat_times_vec(A, x));
}





