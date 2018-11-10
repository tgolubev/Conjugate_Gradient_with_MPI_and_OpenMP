#include "conj_grad_solve.hpp"
#include "IO.hpp"

using vec = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)


// Matrix times vector
vec mat_times_vec(const std::vector<vec> &sub_A, const vec &v)
{    
    
   // NOTE: when using MPI with > 1 proc, A will be only a sub-matrix (a subset of rows) of the full matrix
   // since we are 1D decomposing the matrix by rows
   
   int n = sub_A.size();
   vec sub_c(n);       // for > 1proc, this will be only a sub-vector (portion of the full vector) c, b/c we are multiplying only part of A in each proc.
   for (int i = 0; i < n; i++) // loop over the rows of the sub matrix
       sub_c[i] = dot_product(sub_A[i], v);  // dot product of ith row of sub_A with the vector v       
   
   return sub_c;
}

// Linear combination of vectors
vec vec_lin_combo(double a, const vec &u, double b, const vec &v)
{
   int n = u.size();
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


// performs a reduction over the sub-vectors which are passed to it... All_Reduce broadcasts the value to all procs
double mpi_dot_product(const vec &sub_u, const vec &sub_v) // need to pass it the buffer where to keep the result
{
   double product;
   double sub_prod = inner_product(sub_u.begin(), sub_u.end(), sub_v.begin(), 0.0); // last argument is initial value of the sum of products
   // sub_prod works!
   //std::cout << "sub_prod" << sub_prod << std::endl;
   // LATER the inner_product will be replaced by CUDA to further divide up the work
   
   // do a reduction over sub_prod to get the total dot product
   MPI_Allreduce(&sub_prod, &product, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   return product;
}

// vector norm
double vector_norm(const vec &v)
{
   return sqrt(dot_product(v, v));
}

double mpi_vector_norm(const vec &sub_v)
{
   double norm_r;
   double sub_prod = inner_product(sub_v.begin(), sub_v.end(), sub_v.begin(), 0.0);
   MPI_Allreduce(&sub_prod, &norm_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   
   norm_r = sqrt(norm_r);

   return norm_r;
}


vec conj_grad_solver(const mat &A, const vec &b)
{
    
   // NOTE: when using MPI with > 1 proc, A will be only a sub-matrix (a subset of rows) of the full matrix
   // since we are 1D decomposing the matrix by rows
    // b will be the full vector
   int nprocs, rank;
   MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank (MPI_COMM_WORLD, &rank);


   // Domain decomposition
   int row_cnt[nprocs];  // to keep track of # of rows in each rank, when not evenly divisible, this can be different
   int row_disp[nprocs];  // displacement from start of vector, needed for gatherv
   int m = b.size();

   for (int i = 0; i < nprocs; i++) {
       row_cnt[i] = m/nprocs;   //NOTE: later need to take care of possiblility that m may not be evenly divisible by nprocs
       row_disp[i] = i * m/nprocs;  // NOTE: this is assuming that rank 0 is also doing work. LATER I MIGHT MAKE rank 0 be a master and only doing initializations.
   }

   mat sub_A(m/nprocs, std::vector<double> (A[0].size()));  // note: this is the correct way to initialize a vector of vectors.
   // note: A[0].size() gives # of elements in 1st row = # of columns
   // this is #rows/nprocs for the row #, and column # is same as in A
   //NOTE: WILL LATER NEED TO ACCOUNT FOR POSSIBILITY THAT ROWS DON'T EVENLY DIVIDE INTO THE NPROCS!!
   for (int i = 0; i < m/nprocs; i++)
       for (int j = 0; j < m/nprocs; j++)
         sub_A[i][j] = A[rank * m/nprocs + i][j];

   std::cout << "sub_A " << std::endl;
   print(sub_A);
   std::cout << "A " << std::endl;
   print(A);

   // with 1 proc, subA and A are correct here...

   double tolerance = 1.0e-10;

   int n = sub_A.size();
   vec sub_x(n/nprocs); // iniitalize a vector to store the solution subvector
   vec x(n);
   // we want a decomposed r

   vec sub_r(m/nprocs);  
   for (int i = 0; i < m; i++)
        sub_r[i] = b[m*rank + i];

   
   // BUT NEED TO TAKE INTO  ACCOUNT THAT b could be not evenenly  divisible by nprocs!
   // MAYBE DEAL WITH THIS ISSUE LATER
   
   //---------------------------------------------------------------------------------------------------------
   vec p = b;  //we want a full p
   vec sub_p = sub_r; //also we want a sub_p, decomposed to be able to split up the work
   vec r_old;
   vec sub_a_times_p;
   // SEEMS TO BLOW UP ON THE SECOND ITERATION!!!
   for (int i = 0; i < n; i++) {  // this loop must be serial b/c is iterations of conj_grad 
       // THIS MIGHT MAKE MORE SENSE TO BE i< max_iter, but I guess can use n = a.size() for max iter?
      r_old = sub_r;                                         // Store previous residual
      // here we want to use a split r!
      
      sub_a_times_p = mat_times_vec(sub_A, p);  //split up with MPI and then finer parallelize with CUDA-->
      //means have multiple GPU's --> 1 corresponding to each CPU
      // NOTE: a_times_p will only be a part of the full A*p for >1proc
      
      // SUB R IS CORRECT HERE, = b in the 1st iter

      // mpi_dot_product will perform a reduction over the sub vectors to give back the full vector!
      double alpha = mpi_dot_product(sub_r, sub_r)/std::max(mpi_dot_product(sub_p, sub_a_times_p), tolerance);

      
      // Next estimate of solution  // this is like saxpy: use CUDA
      sub_x = vec_lin_combo(1.0, sub_x, alpha, sub_p );             // note: sub_x is a buffer where the solution goes
      // this will also only return a sub vector of x...
       
      std::cout << "sub_x" << std::endl;
      print(sub_x);

      sub_r = vec_lin_combo(1.0, sub_r, -alpha, sub_a_times_p);          // Residual                   // again like saxpy: use CUDA

      std::cout << "sub_r" << std::endl;
      print(sub_r);


      // Convergence test
      if (mpi_vector_norm(sub_r) < tolerance)  // vector norm needs to use a all reduce!
          break;

      std::cout <<  "sub r norm " << mpi_vector_norm(sub_r) << std::endl;
      
      double beta = mpi_dot_product(sub_r, sub_r)/std::max(mpi_dot_product(r_old, r_old), tolerance);
      std::cout << "beta" << beta << std::endl;

      sub_p = vec_lin_combo(1.0, sub_r, beta, sub_p);             // Next gradient                     // again like saxpy: use CUDA

      std::cout << "sub_p" << std::endl;
      print(sub_p);

      // WE NEED TO UPDATE THE p (full vector)  value  through a gather!! for the next iteration b/c it's needed in mat_times_vec
      MPI_Gatherv(&sub_p.front(), row_cnt[rank], MPI_DOUBLE, &p.front(), row_cnt, row_disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   }

   // need a final gather to get back to full x...
   MPI_Gatherv(&sub_x.front(), row_cnt[rank], MPI_DOUBLE, &x.front(), row_cnt, row_disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   return x;
}
