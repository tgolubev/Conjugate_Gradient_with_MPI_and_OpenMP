#include "IO.hpp"

using vec    = std::vector<double>;         // vector
using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)


void write_solution_hdf5(const vec &solution, const int n)
{

    hid_t       file_id;   /* file identifier */
    herr_t      status;
    double dset_data[n];

    /* Create a new file using default properties. */
    file_id = H5Fcreate("results.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Create a dataspace
    hsize_t dims[0];
    dims[0] = n;
    hid_t dataspace_id = H5Screate_simple(0, dims, NULL);  //(rank, size of each dimesnion of dataset, max dims)

    /* Initialize the dataset. */
    // copy the data into C-style array, from std::vector
       for (int i = 0; i < n; i++)
            dset_data[i] = solution[i];

       // LETS FIRST CREATE THE HDF5 FILE WITH DATASET IN MATLAB, then I will just open the existing fiel and dataset with this function and write to it!

   /*
    // In future, I WILL CREATE THE HDF5 FILE in Matlab when I generate the matrix to be solved...,
    // then will open it and add the solution to the dataset
    // Open an existing file.
      file_id = H5Fopen(FILE, H5F_ACC_RDWR, H5P_DEFAULT);

      // Open an existing dataset.
      dataset_id = H5Dopen(file_id, "/dset");
      */


      /* Write the dataset. */
      //status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset_data);

     /*
      // read example
      status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       dset_data);
                       */


    /*
    // now create a real dataset
    hid_t dataset_id = H5Dcreate(file_id, "/dset", HSD_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // write data to the file
    status = H5Dwrite(dataset_id, HST_NATIVE_INT, HSS_ALL, HSS_ALL, HSP_DEFAULT, solution);

    // close the dataset and dataspace
    status = H5Dclose(dataset_id);
    */
    status = H5Sclose(dataspace_id);
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
   size_t m = A.size();
   size_t n = A[0].size();                      // A is an m x n matrix
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
