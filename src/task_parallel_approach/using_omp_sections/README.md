# Instructions to run on HPCC at MSU

Note: these particular modules are needed in order to be able to use HDF5:

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

    
* NOTE: the number of openmp threads is controlled by `export  OMP_NUM_THREADS=` which is set inside of the makefile.
* Or can also change the num threads environmental variable intereactively by putting into the terminal: `export OMP_NUM_THREADS=4`

* it is good to verify that the # of openmp threads is what you think it is by using an omp threads function call inside the program:
    
   # pragma omp parallel
   {
      printf("Num OpenMP threads: %d\n", omp_get_num_threads());
   }
