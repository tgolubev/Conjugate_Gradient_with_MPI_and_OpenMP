#include "conj_grad_solve.hpp"

using vec = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)


// Matrix times vector
vec mat_times_vec(const std::vector<vec> &A, const vec &v)
{
   int n = A.size();
   vec c(n);
   for (int i = 0; i < n; i++)
       c[i] = dot_product(A[i], v);
   return c;
}

// Linear combination of vectors
vec vec_lin_combo(double a, const vec &u, double b, const vec &v)
{
   int n = size(u);
   vec w(n);
   for (int j = 0; j < n; j++)
       w[j] = a * u[j] + b * v[j];
   return w;
}

// Inner product of u and v

double dot_product(const vec &u, const vec &v)
{
   return inner_product(u.begin(), u.end(), v.begin(), 0.0);
}


// vector norm
double vector_norm(const vec &v)
{
   return sqrt(dot_product(v, v));
}




vec conj_grad_solver(const mat &A, const vec &b)
{
   double tolerance = 1.0e-10;

   int n = A.size();
   vec x(n, 0.0);

   vec r = b;
   vec p = r;
   
   for (int i = 0; i < n; i++) {
      vec r_old = r;                                         // Store previous residual
      vec a_times_p = mat_times_vec(A, p);

      double alpha = dot_product(r, r)/std::max(dot_product(p, a_times_p), tolerance);
      x = vec_lin_combo(1.0, x, alpha, p );                  // Next estimate of solution
      r = vec_lin_combo(1.0, r, -alpha, a_times_p);          // Residual

      // Convergence test
      if (vector_norm(r) < tolerance)
          break;             

      double beta = dot_product(r, r)/std::max(dot_product(r_old, r_old), tolerance);
      p = vec_lin_combo(1.0, r, beta, p);             // Next gradient
   }

   return x;
}
