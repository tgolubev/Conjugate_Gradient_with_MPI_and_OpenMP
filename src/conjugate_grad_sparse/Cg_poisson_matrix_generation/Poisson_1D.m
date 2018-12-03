% 1D poisson equation solving for benchmarking

n = 30000;
h = 1;
h = 1/(n+1);
x = h:h:1-h;


clf
K1D = spdiags(ones(n,1)*[-1 2 -1],-1:1,n,n);  % 1d Poisson matrix


f1D = h^2*ones(n,1);                          % 1d right hand side
% u1D = K1D\f1D;



% loop through the dense_K3D matrix to store the information in
% compressed row storage (CRS) format
% This means we store the values in 1 array, and the column indices in
% another for non-zero elements only and do this by row.

% so i.e. for a row:  1 0 0 2 0 4
% we would store: values = [1 2 4] and colm_indices = [1 4 6]
% This way when we do matrix*vector multiply we can use the
% colm_indices to know where to multiply.., so i..e

% values[1]*vector[colm_indices[1]] + values[2] *
% vector[[colm_indeces[2]] + .....
% THIS IS FOR SQUARE MATRICES

values = zeros(n, 7);
col_indices = zeros(n, 7);

for row = 1:size(K1D, 1)
    non_zero_col_cnt = 0;
    for col = 1:size(K1D, 1)
        if K1D(row, col) ~= 0
            non_zero_col_cnt = non_zero_col_cnt + 1;
            values(row, non_zero_col_cnt) = K1D(row, col);
            col_indices(row, non_zero_col_cnt) = col;
        end
    end
end

values;
col_indices;


% tic
% 
% %     pcg(K3D, f3D, 1e-8);
% 
% toc
initial_guess = zeros(size(K1D,1), 1);

% nOTE: WE MUST TRAnSPOSE THE MATRICES In ORDER FOR THE OUTPUT THE HDF5
% TO MATCH HOW READInG IT In C++!!!

h5create('cg_1D_poisson_sparse.h5','/matrix_values',size(values.'));
h5create('cg_1D_poisson_sparse.h5','/col_indices',size(col_indices.'));
h5create('cg_1D_poisson_sparse.h5','/solution',size(f1D)); % prepare data set for solution
h5create('cg_1D_poisson_sparse.h5','/initial_guess',size(f1D));
h5create('cg_1D_poisson_sparse.h5','/rhs',size(f1D));
h5create('cg_1D_poisson_sparse.h5','/cpu_time',1);  % spot to store the cpu time
h5create('cg_1D_poisson_sparse.h5', '/cpu_per_iter', 1);
h5create('cg_1D_poisson_sparse.h5','/num_iters',1); %store number of iters it took to solve
h5create('cg_1D_poisson_sparse.h5','/tolerance',1);
h5create('cg_1D_poisson_sparse.h5','/error',size(f1D));

h5write('cg_1D_poisson_sparse.h5', '/matrix_values', values.');
h5write('cg_1D_poisson_sparse.h5', '/col_indices', col_indices.');
h5write('cg_1D_poisson_sparse.h5', '/initial_guess',initial_guess);
h5write('cg_1D_poisson_sparse.h5', '/rhs', f1D);


size(K1D)
size(values)
