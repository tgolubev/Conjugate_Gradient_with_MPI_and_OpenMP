#include "incomplete_cholesky.hpp"

using vec = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)

//Reference: Golub, Gene H.; Van Loan, Charles F. (1996), Matrix Computations (3rd ed.), Johns Hopkins Section 10.3.2

// this is the serial version (i.e. can call this from main.cpp before send to CG algorithm
//! This seems to not help the preconditioning, someitmes in matlab makes things even harder to solve, than without it!
void incomplete_cholesky(mat &A)
{
    size_t n = A[0].size();

    for (size_t k = 0; k < n; k++) {
        A[k][k] = sqrt(A[k][k]);
        for (size_t i = k+1; i < n; i++)
            if (A[i][k] != 0.0)
                A[i][k] = A[i][k]/A[k][k];

        for (size_t j = k+1; j < n; j++)
            for (size_t i = j; i < n; i++)
                if (A[i][j] != 0.0)
                    A[i][j] = A[i][j] - A[i][k]*A[j][k];

     }

    for (size_t i = 0; i < n; i++)
         for (size_t j = i+1; j < n; j++)
             A[i][j] = 0;

}

// ref: http://www.jpier.org/PIERB/pierb13/03.08112407.pdf   algorithm 4
//void modified_ichol(mat &A)
//{
//    size_t n = A[0].size();
//    vec w(n);

//    for (size_t j = 0; j < n; j++) {
//        A[j][j] = sqrt(A[j][j]);
//        w[j] = a[j+1][j];
//        for (size_t k = 0; k < j-1; k++)
//            for (size_t i = j+1; i < n; i++)
//                if (A[j][k] != 0.0)
//                    w[i] = w[i]] - A[i][k]*A[j][k];

//        for (size_t i = j+1; i < n; i++)
//            w[i][j] = w[i][j]/A[j][j];
//}
