# Comment lines
# Here we define compiler option, libraries and the target
FLAGS = -O3 -Wall -fopenmp -std=c++11

# Here we make the executable file
SRCS =   incomplete_cholesky.cpp main.cpp IO.cpp conj_grad_solve.cpp 
OBJS = $(subst .cpp,.o,$(SRCS))
all: cg

# Whereas here we create the object file
cg: $(OBJS)  
	mpic++ ${FLAGS} -o cg $(OBJS) # note: must use mpic++ compiler for MPI with cpp files (mpicc for .c)
	export OMP_NUM_THREADS=2  # use this for OpenMP
#	./md

%.o: %.cpp 
	mpic++ $(FLAGS) -c $<



# Clean
clean:
	rm *.o ./cg
