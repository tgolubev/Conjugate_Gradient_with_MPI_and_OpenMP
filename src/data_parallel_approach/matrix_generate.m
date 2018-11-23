% generation of a dense matrix which can be solved with conjugate gradient

% N = 50;
% A = delsq(numgrid('S',N)); %sparse matrix
% b = rand(size(A,1),1);  % use random #'s btw 0 and 1 for rhs

n1 = 1584; % size of square matrix
A = gallery('moler',n1);  %dense matrix
b = sum(A,2);

tic

x = pcg(A, b, 1e-8, 10000);

toc

% cholesky factorization
% A_factorized = ichol(A)

A_dense = full(A);
% A_factorized_dense = full(A_factorized);

initial_guess = zeros(size(A,1), 1);

tic

x = pcg(A_dense, b, 1e-8, 10000);

toc

% save to HDF5 file
h5create('cg.h5','/matrix',size(A_dense));
h5create('cg.h5','/solution',size(x)); % prepare data set for solution
h5create('cg.h5','/initial_guess',size(x));
h5create('cg.h5','/rhs',size(x));
h5create('cg.h5','/cpu_time',1);  % spot to store the cpu time
h5create('cg.h5', '/cpu_per_iter', 1);
h5create('cg.h5','/num_iters',1); %store number of iters it took to solve
h5create('cg.h5','/tolerance',1);
h5create('cg.h5','/error',size(x));

h5write('cg.h5', '/matrix', A_dense);
h5write('cg.h5', '/initial_guess',initial_guess);
h5write('cg.h5', '/rhs', b);


% save to file: regular text file
% format long
% save('matrix.txt', 'A_dense', '-ascii','-double');
% % save('matrix_factorized.txt', 'A_factorized_dense', '-ascii','-double');
% save('rhs.txt', 'b', '-ascii','-double');
% save('solution.txt', 'x','-ascii','-double');
% %save('initial_guess.txt', 'initial_guess','-ascii','-double');