#include "IO.hpp"

using vec    = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)

mat read_sub_mat_hdf5(const char *filename, const char *mat_dataset_name, const int n, const int num_cols)
{

    int nprocs, rank;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    hid_t       file_id, dataset_id;   // identifiers
    herr_t      status;
    double dset_data[n][num_cols];  // a buffer for reading in the matrix
    mat matrix(n, std::vector<double>(num_cols));

    // I WILL CREATE THE HDF5 FILE in Matlab when I generate the matrix to be solved...,
    // then will open it and add the solution to the dataset
    // Open an existing file.
    file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

    //-----------------------------------------------------------------------------
    // Open an existing dataset
    dataset_id = H5Dopen(file_id, mat_dataset_name, H5P_DEFAULT);

    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);

    mat sub_A(n/static_cast<size_t>(nprocs), std::vector<double> (num_cols));  // note: this is the correct way to initialize a vector of vectors.
    for (size_t i = 0; i < n/static_cast<size_t>(nprocs); i++)
        for (size_t j = 0; j < num_cols; j++) {
            sub_A[i][j] = dset_data[static_cast<size_t>(rank) * n/static_cast<size_t>(nprocs) + i][j];
            std::cout << i << ", j = " << j << "suba[i][j] = " << sub_A[i][j] << std::endl;
        }

    return sub_A;

}

// full matrix is needed for verification
mat read_mat_hdf5(const char *filename, const char *mat_dataset_name, const int n, const int num_cols)
{

    int nprocs, rank;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    hid_t       file_id, dataset_id;   // identifiers
    herr_t      status;
    double dset_data[n][num_cols];  // a buffer for reading in the matrix
    mat matrix(n, std::vector<double>(num_cols));

    // I WILL CREATE THE HDF5 FILE in Matlab when I generate the matrix to be solved...,
    // then will open it and add the solution to the dataset
    // Open an existing file.
    file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

    //-----------------------------------------------------------------------------
    // Open an existing dataset
    dataset_id = H5Dopen(file_id, mat_dataset_name, H5P_DEFAULT);

    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);

    // fill the matrix
    for (int i = 0; i < n; i++)
        for (int j = 0; j < num_cols; j++)
            matrix[i][j] = dset_data[i][j];

    return matrix;

}



vec read_vec_hdf5(const char *filename, const char *vec_dataset_name, const int n)
{

    hid_t       file_id, dataset_id;   // identifiers
    herr_t      status;
    double dset_data[n];  // a buffer for reading in the matrix
    vec vector(n);

    // I WILL CREATE THE HDF5 FILE in Matlab when I generate the matrix to be solved...,
    // then will open it and add the solution to the dataset
    // Open an existing file.
    file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

    //-----------------------------------------------------------------------------
    // Open an existing dataset
    dataset_id = H5Dopen(file_id, vec_dataset_name, H5P_DEFAULT);

    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);

    // fill the matrix
    for (int i = 0; i < n; i++) {
            vector[i] = dset_data[i];
    }

    return vector;
}




void write_results_hdf5(const char *filename, const vec &solution, const vec &error, const int n, const double cpu_time, const double cpu_time_per_iter, const double tolerance, const int total_iters)
{

    hid_t       file_id, dataset_id;   // identifiers
    herr_t      status;
    double dset_data[n];

    // I WILL CREATE THE HDF5 FILE in Matlab when I generate the matrix to be solved...,
    // then will open it and add the solution to the dataset
    // Open an existing file.
    file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);

    //-----------------------------------------------------------------------------
    // Open an existing dataset.
    dataset_id = H5Dopen(file_id, "/solution", H5P_DEFAULT);

    // Initialize the dataset.
    // copy the data into C-style array, from std::vector
    for (int i = 0; i < n; i++)
        dset_data[i] = solution[i];

    // Write the dataset
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);

    //-----------------------------------------------------------------------------
    // Open another dataset
    dataset_id = H5Dopen(file_id, "/error", H5P_DEFAULT);

    for (int i = 0; i < n; i++)
        dset_data[i] = error[i];

    // Write the dataset
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);

    //-----------------------------------------------------------------------------

    // Open another dataset
    dataset_id = H5Dopen(file_id, "/cpu_time", H5P_DEFAULT);

    // Write the dataset
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &cpu_time);
    // note: hdf5 requires parameters to be passed as references

    //-----------------------------------------------------------------------------

    // Open another dataset
    dataset_id = H5Dopen(file_id, "/cpu_per_iter", H5P_DEFAULT);

    // Write the dataset
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &cpu_time_per_iter);

    //-----------------------------------------------------------------------------


    // Open another dataset
    dataset_id = H5Dopen(file_id, "/tolerance", H5P_DEFAULT);

    // Write the dataset
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &tolerance);

    //-----------------------------------------------------------------------------

    // Open another dataset
    dataset_id = H5Dopen(file_id, "/num_iters", H5P_DEFAULT);

    // Write the dataset
    status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &total_iters);

    //-----------------------------------------------------------------------------
    // close the dataset and file
    status = H5Dclose(dataset_id);
    status = H5Fclose(file_id);


    /* EXAMPLE CODE IS HERE
     *
     * /*
 *   Writing and reading an existing dataset.
 */

//#include "hdf5.h"
//#define FILE "dset.h5"

//main() {

//   hid_t       file_id, dataset_id;  /* identifiers */
//   herr_t      status;
//   int         i, j, dset_data[4][6];

//   /* Initialize the dataset. */
//   for (i = 0; i < 4; i++)
//      for (j = 0; j < 6; j++)
//         dset_data[i][j] = i * 6 + j + 1;

//   /* Open an existing file. */
//   file_id = H5Fopen(FILE, H5F_ACC_RDWR, H5P_DEFAULT);

//   /* Open an existing dataset. */
//   dataset_id = H5Dopen(file_id, "/dset");

//   /* Write the dataset. */
//   status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
//                     dset_data);

//   status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
//                    dset_data);

//   /* Close the dataset. */
//   status = H5Dclose(dataset_id);

//   /* Close the file. */
//   status = H5Fclose(file_id);

//   */
}






void print(const vec &V)
{

   size_t n = V.size();
   for (size_t i = 0; i < n; i++)
   {
      double x = V[i];   
      std::cout << std::fixed << std::setprecision(10) << x << '\n';
   }
   std::cout<< '\n';
}


void print(const mat &A)
{
   size_t m = A.size(); // this gives # of rows
   size_t n = A[0].size();    // gives # of columns                   // A is an m x n matrix
   for (size_t i = 0; i < m; i++)
   {
      for (size_t j = 0; j < n; j++)
      {
         double x = A[i][j];   
         std::cout << std::fixed << std::setw(10) << std::setprecision(5) << x;
      }
      std::cout << '\n';
   }
}

mat read_matrix(const unsigned int n, std::string filename)
{
    std::ifstream matrix_file;
    matrix_file.open(filename);

    // set the correct size
    mat input_mat(n, std::vector<double>(n)); // recall that mat is a vector of vectors, hence the tricky notation for initialization

    // read in the values
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            matrix_file >> input_mat[i][j];

    return input_mat;
}

vec read_vector(const unsigned int n, std::string filename)
{
    std::ifstream vector_file;
    vector_file.open(filename);

    vec input_vec(n);

    for (size_t i = 0; i < n; i++)
        vector_file >> input_vec[i];

    return input_vec;
}
