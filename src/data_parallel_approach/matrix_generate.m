% generation of a dense matrix which can be solved with conjugate gradient

N = 20;
A = delsq(numgrid('S',N)); %sparse matrix
b = zeros(size(A,1),1);
b(6) = 1;

%n1 = 20; % size of square matrix
%A = gallery('moler',n1);  %dense matrix
%b = sum(A,2);
% 
% tic
% 
% x = pcg(A, b, 1e-6, 10000);
% 
% toc

% cholesky factorization
% A_factorized = ichol(A)

A_dense = full(A);
% A_factorized_dense = full(A_factorized);

%initial_guess = zeros(324, 1);

tic

x = pcg(A_dense, b, 1e-8, 10000);

toc

% save to HDF5 file
h5create('cg.h5','/matrix',size(A_dense));
h5write('cg.h5', '/matrix', A_dense);


% save to file: regular text file
format long
save('matrix.txt', 'A_dense', '-ascii','-double');
% save('matrix_factorized.txt', 'A_factorized_dense', '-ascii','-double');
save('rhs.txt', 'b', '-ascii','-double');
save('solution.txt', 'x','-ascii','-double');
%save('initial_guess.txt', 'initial_guess','-ascii','-double');