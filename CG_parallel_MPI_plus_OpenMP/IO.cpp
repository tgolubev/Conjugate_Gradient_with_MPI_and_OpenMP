
#include "IO.hpp"

using vec    = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)

void print(const vec &V)
{

   int n = V.size();           
   for (int i = 0; i < n; i++)
   {
      double x = V[i];   
      std::cout << x << '\n';
   }
   std::cout<< '\n';
}


void print(const mat &A)
{
   int m = A.size();
   int n = A[0].size();                      // A is an m x n matrix
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < n; j++)
      {
         double x = A[i][j];   
         std::cout << x << '\t';
      }
      std::cout << '\n';
   }
}
