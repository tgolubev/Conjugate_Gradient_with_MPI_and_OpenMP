
#include "IO.hpp"

using vec    = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)

void print(const vec &V)
{

   size_t n = V.size();
   for (size_t i = 0; i < n; i++)
   {
      double x = V[i];   
      std::cout << x << '\n';
   }
   std::cout<< '\n';
}


void print(const mat &A)
{
   size_t m = A.size();
   size_t n = A[0].size();                      // A is an m x n matrix
   for (size_t i = 0; i < m; i++)
   {
      for (size_t j = 0; j < n; j++)
      {
         double x = A[i][j];   
         std::cout << x << '\t';
      }
      std::cout << '\n';
   }
}

mat read_matrix(const unsigned int n)
{
    std::ifstream matrix_file;
    matrix_file.open("matrix.txt");

    // set the correct size
    mat input_mat(n, std::vector<double>(n)); // recall that mat is a vector of vectors, hence the tricky notation for initialization

    // read in the values
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            matrix_file >> input_mat[i][j];

    return input_mat;

}
