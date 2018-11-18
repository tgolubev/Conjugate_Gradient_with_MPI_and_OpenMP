# Instructions to run on HPCC at MSU

    module load OpenMPI
    make
    mpirun -n [number of distributed procs for MPI] ./cg

    
* NOTE: the number of openmp threads is controlled by `export  OMP_NUM_THREADS=` which is set inside of the makefile.
