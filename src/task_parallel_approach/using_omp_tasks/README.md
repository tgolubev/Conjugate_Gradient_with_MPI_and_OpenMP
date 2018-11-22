# Instructions to run on HPCC at MSU

To compile cpp file using hdf5:

    h5c++ [cpp src file name]

Note: these particular modules are needed in order to be able to use HDF5:

    module purge
    module load powertools
    module load intel/2018a
    module load OpenMPI/2.1.2
    module load HDF5/1.8.16
    alias devjob='salloc -n 4 --time 1:30:00'  // request an interactive job. This is important! Otherwise it will not always get the threads you ask for and have 
                                                //unreliable timings
    make
    mpirun -n [number of distributed procs for MPI] ./cg
    
If do not need HDF5, can also use:
    module load GNU/7.3.0-2.30
    module load OpenMPI/3.1.1

## Reading the HDF5 file:
The file is pre-generated using Matlab and filled with a matrix, right hand side, and initial guess to be used by the C++ parallelized CG algorithm.
The CG code will add the solution to this file.

In Linus terminal:
     h5ls cg.h5  // will list the directories inside the file
     
     h5ls -d cg.h5/[name of dataset]  // will display the contents (values) of a particular data set


    
* NOTE: the number of openmp threads is controlled by `export  OMP_NUM_THREADS=` which is set inside of the makefile.
* Or can also change the num threads environmental variable intereactively by putting into the terminal: `export OMP_NUM_THREADS=4`

* it is good to verify that the # of openmp threads is what you think it is by using an omp threads function call inside the program:
    
   # pragma omp parallel
   {
      printf("Num OpenMP threads: %d\n", omp_get_num_threads());
   }
