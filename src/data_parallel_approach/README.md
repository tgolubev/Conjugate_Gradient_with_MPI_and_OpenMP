# Instructions to run on HPCC at MSU

    module load OpenMPI
    alias devjob='salloc -n 4 --time 1:30:00'  // request an interactive job. This is important! Otherwise it will not always get the threads you ask for and have 
                                                //unreliable timings
    make
    mpirun -n [number of distributed procs for MPI] ./cg

    
* NOTE: the number of openmp threads is controlled by `export  OMP_NUM_THREADS=` which is set inside of the makefile.
* Or can also change the num threads environmental variable intereactively by putting into the terminal: `export OMP_NUM_THREADS=4`

* it is good to verify that the # of openmp threads is what you think it is by using an omp threads function call inside the program:
    
   # pragma omp parallel
   {
      printf("Num OpenMP threads: %d\n", omp_get_num_threads());
   }
