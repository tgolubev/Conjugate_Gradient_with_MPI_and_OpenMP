% generation of a dense matrix which can be solved with conjugate gradient

n1 = 500; % size of square matrix

A = gallery('moler',n1);  

b = sum(A,2);

tic

x = pcg(A, b)

toc

% save to file
format long
save('matrix.txt', 'A', '-ascii','-double');
save('rhs.txt', 'b', '-ascii','-double');
save('solution.txt', 'x','-ascii','-double');