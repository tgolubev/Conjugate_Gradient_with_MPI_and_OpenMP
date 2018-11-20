#include "conj_grad_solve.hpp"
#include "IO.hpp"
#include <omp.h>

using vec = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)



// Matrix times vector
void mat_times_vec(const std::vector<vec> &sub_A, const vec &v, vec &result)
{    
    
   // NOTE: when using MPI with > 1 proc, A will be only a sub-matrix (a subset of rows) of the full matrix
   // since we are 1D decomposing the matrix by rows
   
   size_t sub_size = sub_A.size();


#pragma omp parallel for // this is dividing the work among the rows...
   for (size_t i = 0; i < sub_size; i++) // loop over the rows of the sub matrix
       result[i] = dot_product(sub_A[i], v);  // dot product of ith row of sub_A with the vector v

}

// Linear combination of vectors
void vec_lin_combo(double a, const vec &u, double b, const vec &v, vec &result)
{
   size_t n = u.size();

   //#pragma omp parallel for  //NOTE: IT IS FASTER WITHOUT THIS PRAGMA!!--> THE OPERATION ISN'T cpu intensive enough so
   // overhead of creating threads is larger than operation
   for (size_t j = 0; j < n; j++)
       result[j] = a * u[j] + b * v[j];

}


// Inner product of u and v
double dot_product(const vec &u, const vec &v)
{
   return __gnu_parallel::inner_product(u.begin(), u.end(), v.begin(), 0.0);  // leave this alone b/c is used within an already multi-threaded region
}


// performs a reduction over the sub-vectors which are passed to it... All_Reduce broadcasts the value to all procs
double mpi_dot_product(const vec &sub_u, const vec &sub_v) // need to pass it the buffer where to keep the result
{
   double product;
   size_t length = sub_u.size();

   // trying to parallelize this made it slower (with 10 threads, 2 procs, than w/o the parallelization!)
   // THAT MIGHT BE B/C THE INNER PRODUCT is a very efficient implementation!! which is hard to beat.
   //double sub_prod = 0.0;
   /*#pragma omp parallel for reduction(+:sub_prod)
   for (size_t i = 0; i < length; i++) {
       sub_prod += sub_u[i] * sub_v[i];
   }*/
   double sub_prod = __gnu_parallel::inner_product(sub_u.begin(), sub_u.end(), sub_v.begin(), 0.0); // last argument is initial value of the sum of products
   // using gnu_parallel instead of regular inner_product didin't really speed things up but didint' slow them down either.

   // sub_prod works!
   //std::cout << "sub_prod" << sub_prod << std::endl;
   
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
   double sub_prod = __gnu_parallel::inner_product(sub_v.begin(), sub_v.end(), sub_v.begin(), 0.0);
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
   size_t m = b.size();

   for (int i = 0; i < nprocs; i++) {
       row_cnt[i] = m/nprocs;   //NOTE: later need to take care of possiblility that m may not be evenly divisible by nprocs
       row_disp[i] = i * m/nprocs;  // NOTE: this is assuming that rank 0 is also doing work. LATER I MIGHT MAKE rank 0 be a master and only doing initializations.
   }

   mat sub_A(m/static_cast<size_t>(nprocs), std::vector<double> (A[0].size()));  // note: this is the correct way to initialize a vector of vectors.
   // note: A[0].size() gives # of elements in 1st row = # of columns
   // this is #rows/nprocs for the row #, and column # is same as in A
   //NOTE: WILL LATER NEED TO ACCOUNT FOR POSSIBILITY THAT ROWS DON'T EVENLY DIVIDE INTO THE NPROCS!!
   for (size_t i = 0; i < m/static_cast<size_t>(nprocs); i++)
       for (size_t j = 0; j < A[0].size(); j++)
         sub_A[i][j] = A[static_cast<size_t>(rank) * m/static_cast<size_t>(nprocs) + i][j];

   double tolerance = 1.0e-5; // seems can't converge to very low tolerance

   size_t n = A.size();
   vec sub_x(n/static_cast<size_t>(nprocs)); // iniitalize a vector to store the solution subvector
   vec x(n);
   // we want a decomposed r

   vec sub_r(m/static_cast<size_t>(nprocs));
   for (size_t i = 0; i < m/static_cast<size_t>(nprocs); i++)
        sub_r[i] = b[(m/static_cast<size_t>(nprocs))*static_cast<size_t>(rank) + i];


   /*
   if(rank  == 0) {
       std::cout << "sub_A " << std::endl;
       print(sub_A);
       std::cout << "A " << std::endl;
       print(A);
       std::cout << "sub_r " << std::endl;
       print(sub_r);
    }
    */
   
   // BUT NEED TO TAKE INTO  ACCOUNT THAT b could be not evenenly  divisible by nprocs!
   // MAYBE DEAL WITH THIS ISSUE LATER
   
   //---------------------------------------------------------------------------------------------------------
   vec p = b;  //we want a full p
   vec sub_p = sub_r; //also we want a sub_p, decomposed to be able to split up the work
   vec r_old;
   vec result(n/static_cast<size_t>(nprocs));
   vec sub_a_times_p(n/static_cast<size_t>(nprocs));
   int max_iter = 100000;

   double sub_r_sqrd, sub_p_by_ap, norm_sub_r;

   for (int i = 0; i < max_iter; i++) {  // this loop must be serial b/c is iterations of conj_grad
       // note: make sure matrix is big enough for thenumber of processors you are using!

      r_old = sub_r;                                         // Store previous residual

      mat_times_vec(sub_A, p, sub_a_times_p);  //split up with MPI and then finer parallelize with openmp

      double sub_sub_r_sqrd, sub_sub_p_by_ap;


// using tasks is better than sections b/c tasks will allow for a thread to omove on to a new task when finished with current one

#pragma omp parallel
#pragma omp single // single works, but when add tasks, things fail. // single says to do only each task once
{
        #pragma omp task
        {
            sub_sub_r_sqrd = std::inner_product(sub_r.begin(), sub_r.end(), sub_r.begin(), 0.0);
        }
        #pragma omp task
        {
            sub_sub_p_by_ap = std::inner_product(sub_p.begin(), sub_p.end(), sub_a_times_p.begin(), 0.0);
        }

        #pragma omp taskwait

        // THIS WORKS!! --> just the Allreduce can't be inside the tasks!!, needs to be done after a synchornization, taskwait!!
          //i.e. the tasks need to complete before do the reduce!!

        // mpi reduce maybe can't be inside the threads!!..., b/c a single thread doesn't have memory access to all the different procs!!!!!!
        // mpi_dot_product will perform a reduction over the sub vectors to give back the full vector!
        MPI_Allreduce(&sub_sub_r_sqrd, &sub_r_sqrd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&sub_sub_p_by_ap, &sub_p_by_ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        #pragma omp task default(shared) depend(out: sub_r_sqrd)  // need to specify the variables to be shared!
           sub_r_sqrd = mpi_dot_product(sub_r, sub_r);

        #pragma omp task default(shared)
            sub_p_by_ap = mpi_dot_product(sub_p, sub_a_times_p);

        #pragma omp taskwait // this is a barrier, since need to have both above tasks done before next step

}



      double alpha = sub_r_sqrd/std::max(sub_p_by_ap, tolerance);

      // Next estimate of solution

      //#pragma omp task
      //{
        vec_lin_combo(1.0, sub_x, alpha, sub_p, result);
        sub_x = result;
      //}
      //#pragma omp task depend(out:sub_r)
      //{
          vec_lin_combo(1.0, sub_r, -alpha, sub_a_times_p, result);
          sub_r = result;
      //}
      //#pragma omp task depend(in: sub_r) depend(out: sub_r_sqrd)
        sub_r_sqrd = mpi_dot_product(sub_r, sub_r);



// recall that we can't have a 'break' within an openmp parallel region, so end it here then all threads are merged, and the convergence is checked

      // Convergence test
        if (sqrt(sub_r_sqrd) < tolerance) { // norm is just sqrt(dot product so don't need to use a separate norm fnc) // vector norm needs to use a all reduce!
             std:: cout << "Converged at iter = " << i << std::endl;
             break;
         }


      double beta = sub_r_sqrd/std::max(mpi_dot_product(r_old, r_old), tolerance);

      vec_lin_combo(1.0, sub_r, beta, sub_p, result);             // Next gradient
      sub_p = result;



      // WE NEED TO UPDATE THE p (full vector)  value  through a gather!! for the next iteration b/c it's needed in mat_times_vec
      MPI_Allgatherv(&sub_p.front(), row_cnt[rank], MPI_DOUBLE, &p.front(), row_cnt, row_disp, MPI_DOUBLE, MPI_COMM_WORLD);
      // need an Allgatherv b/c all procs need the p vector

   }

   // need a final gather to get back to full x...
   MPI_Gatherv(&sub_x.front(), row_cnt[rank], MPI_DOUBLE, &x.front(), row_cnt, row_disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   return x;
}
