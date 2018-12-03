%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Poisson matrix generation from 3D Drift Diffusion for Holes with Finite
%  Differences Model
%            
%           This is a portion of my research code which generates a 3D
%           Poisson matrix which describes a real physical system. I am
%           using this to test my conjugate gradient algorithm with a
%           realistic matrix.
%
%                    By: Timofey Golubev (2018.06.29)
%
%
%   Boundary conditions for Poisson equation are:
%
%     -a fixed voltage at (x,0) and (x, Nz) defined by V_bottomBC
%      and V_topBC which are defining the  electrodes
%
%    -insulating boundary conditions: V(0,y,z) = V(1,y,z) and
%     V(N+1,y,z) = V(N,y,z) (N is the last INTERIOR mesh point).
%     so the potential at the boundary is assumed to be the same as just inside
%     the boundary. Gradient of potential normal to these boundaries is 0.
%    V(x,0,z) = V(x,1,z) and V(x,N+1,z) = V(x,N,z)
%
%   Matrix equation is AV*V = bV is sparse matrices
%   (generated using spdiag), for the Poisson equations.
%   V is the solution for electric potential
%   bV is the rhs of Poisson eqn which contains the charge densities and boundary conditions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;
global num_cell N num_elements Vt N_dos p_topBC p_bottomBC CV Cp 
global V_bottomBC V_topBC G_max

%% Physical Constants

q =  1.60217646*10^-19;         %elementary charge (C)
kb = 1.3806503*10^-23;          %Boltzmann const. (J/k)
T = 296.;                       %temperature (K)
epsilon_0 =  8.85418782*10^-12; %F/m
Vt = (kb*T)/q;

%% Simulation Setup

%Voltage sweep loop
Va_min = -0.5;            %volts
Va_max = -0.45;
increment = 0.01;         %by which to increase V
num_V = floor((Va_max-Va_min)/increment)+1;   %number of V points

%Simulation parameters
w_eq = 0.2;               %linear mixing factor for 1st convergence (0 applied voltage, no generation equilibrium case)
w_i = 0.2;                 %starting linear mixing factor for Va_min (will be auto decreased if convergence not reached)
tolerance = 5*10^-12;        %error tolerance
tolerance_i =  5*10^-12;     %initial error tolerance, will be increased if can't converge to this level

%% System Setup
L = 5.0000001e-9;     %there's some integer rounding issue, so use this .0000001
dx = 1e-9;                        %mesh size
num_cell = floor(L/dx);
N = num_cell -1;       %number of INTERIOR mesh points (total mesh pts = num_cell +1 b/c matlab indixes from 1)
num_elements = N*(N+1)^2;  %NOTE: this will specify number of elements in the solution vector V which = N*(N+1)^2 b/c in x, y direction we include the right bc for Pbc's so have N+1. In k we have just N

%Electronic density of states of holes and electrons
N_VB = 10^24;         %density of states in valence band (holes)
N_CB = 10^24;         %density of states in conduction bands (electrons)
E_gap = 1.5;          %bandgap of the active layer(in eV)
N_dos = 10^24.;       %scaling factor helps CV be on order of 1

%injection barriers
inj_a = 0.2;	%at anode
inj_c = 0.1;	%at cathode

%work functions of anode and cathode
WF_anode = 4.8;
WF_cathode = 3.7;

Vbi = WF_anode - WF_cathode +inj_a +inj_c;  %built-in field

G_max = 4*10^27;

%% Define matrices of system parameters

tic

%Preallocate vectors and matrices
fullV = zeros(N+2, N+2, N+2);
V_values = zeros(num_V+1,1);

% Relative dielectric constant matrix (can be position dependent)
%Epsilons are defined at 1/2 integer points, so epsilons inside
%the cells, not at cell boundaries
%will use indexing: i + 1/2 is defined as i+1 for the index
epsilon = 3.0*ones(num_cell+2, num_cell +2, num_cell +2);
for i =  1:num_cell+1   %go extra +1 so matches size with Bernoulli fnc's which multiply by
    for j = 1:num_cell+1
        for k = 1:num_cell+1
            epsilon_avged.eps_X_avg(i,j,k) = (epsilon(i,j,k) + epsilon(i,j+1,k) + epsilon(i,j,k+1) + epsilon(i,j+1,k+1))./4.;
            epsilon_avged.eps_Y_avg(i,j,k) = (epsilon(i,j,k) + epsilon(i+1,j,k) + epsilon(i,j,k+1) + epsilon(i+1,j,k+1))./4.;
            epsilon_avged.eps_Z_avg(i,j,k) = (epsilon(i,j,k) + epsilon(i+1,j,k) + epsilon(i,j+1,k) + epsilon(i+1,j+1,k))./4.;
            
            
            %             n_mob_avged.n_mob_X_avg(i,j,k) = (n_mob(i,j,k) + n_mob(i,j+1,k) + n_mob(i,j,k+1) + n_mob(i,j+1,k+1))./4.;
            %             n_mob_avged.n_mob_Y_avg(i,j,k) = (n_mob(i,j,k) + n_mob(i+1,j,k) + n_mob(i,j,k+1) + n_mob(i+1,j,k+1))./4.;
            %             n_mob_avged.n_mob_Z_avg(i,j,k) = (n_mob(i,j,k) + n_mob(i+1,j,k) + n_mob(i,j+1,k) + n_mob(i+1,j+1,k))./4.;
            %add num_cell+2 values for i to account for extra bndry pt
            epsilon_avged.eps_X_avg(num_cell+2,j,k) = epsilon_avged.eps_X_avg(1,j,k);
            epsilon_avged.eps_Y_avg(num_cell+2,j,k) = epsilon_avged.eps_Y_avg(1,j,k);
            epsilon_avged.eps_Z_avg(num_cell+2,j,k) = epsilon_avged.eps_Z_avg(1,j,k);
            
            %add num_cell+2 values for j to account for extra bndry pt
            epsilon_avged.eps_X_avg(i,num_cell+2,k) = epsilon_avged.eps_X_avg(i,1,k);
            epsilon_avged.eps_Y_avg(i,num_cell+2,k) = epsilon_avged.eps_Y_avg(i,1,k);
            epsilon_avged.eps_Z_avg(i,num_cell+2,k) = epsilon_avged.eps_Z_avg(i,1,k);
        end
    end
end
%NOTE: NEED 1 MORE at num_cell+2 FOR X AND Y values only
%-->this is the wrap around average for
%PBC's! Takes avg of N+1th element with the 0th (defined as 1st element in
%Matlab)
%FOR NOW JUST ASSUME (REASONABLY) THAT EPSILONS ALONG X, Y BOUNDARIES WILL BE
%UNIFORM.....--> OTHERWISE PBC's don't make ANY SENSE!!!--> SINCE NEED TO
%HAVE CONTINOUS EPSILONG IN THOSE DIRECTIONS to extend them periodically
% so I just added another element to j and k in above loop

%--------------------------------------------------------------------------
%Scaling coefficients
CV = (N_dos*dx^2*q)/(epsilon_0*Vt);    %relative permitivity was moved into the matrix

%% Define Poisson equation boundary conditions and initial conditions

% Initial conditions
V_bottomBC(1:N+2, 1:N+2) = -((Vbi)/(2*Vt)-inj_a/Vt);  %needs to be matrix, so can add i.e. afm tip
V_topBC(1:N+2, 1:N+2) = (Vbi)/(2*Vt)-inj_c/Vt;
diff = (V_topBC(1,1) - V_bottomBC(1,1))/num_cell;
% V(1:N) = V_bottomBC + diff;  %define V's corresponding to 1st subblock here (1st interior row of system)
% index = 0;

%-------------------------------------------------------------------------------------------------
%% Define continuity equn boundary and initial conditions
%these are scaled
% n_bottomBC = N_CB*exp(-(E_gap-inj_a)/Vt)/N_dos;
p_bottomBC(1:N+2,1:N+2) = N_VB*exp(-inj_a/Vt)/N_dos;
% n_topBC = N_CB*exp(-inj_c/Vt)/N_dos;
p_topBC(1:N+2,1:N+2) = N_VB*exp(-(E_gap-inj_c)/Vt)/N_dos;

%define initial conditions as min value of BCs
min_dense = min(p_bottomBC(1,1), p_topBC(1,1));
p = min_dense*ones(num_elements, 1);
% p = n;

%-------------------------------------------------------------------------------------------------

% Set up Poisson matrix equation
AV = SetAV_3D(epsilon_avged);
% [L,U] = lu(AV);  %do and LU factorization here--> since Poisson matrix doesn't change
%this will significantly speed up backslash, on LU factorized matrix
%spy(AV);  %allows to see matrix structure, very useful!

%% Main voltage loop
Va_cnt = 0;
for Va_cnt = 0:0
    tic
    not_converged = false;
    not_cnv_cnt = 0;
    
    %stop the calculation if tolerance becomes too high
    if(tolerance >10^-5)
        break
    end
    
    %1st iteration is to find the equilibrium values (no generation rate)
    if(Va_cnt ==0)
        tolerance = tolerance*10^2;       %relax tolerance for equil convergence
        w = w_eq;                         %use smaller mixing factor for equil convergence
        Va = 0;
        Up = zeros(num_elements, 1);  %better to store as 1D vector
        %         Un= Up;
    end
    if(Va_cnt ==1)
        tolerance = tolerance_i;       %reset tolerance back
        w=w_i;
        %         G = GenerationRate();  %only use it once, since stays constant
    end
    if(Va_cnt >0)
        Va = Va_min+increment*(Va_cnt-1);     %set Va value
        %Va = Va_max-increment*(Va_cnt-1);    %decrease Va by increment in each iteration
    end
    
    %Voltage boundary conditions
    V_bottomBC(1:N+2, 1:N+2) = -((Vbi  -Va)/(2*Vt)-inj_a/Vt);
    V_topBC(1:N+2, 1:N+2) = (Vbi- Va)/(2*Vt) - inj_c/Vt;
    
    
    %% Poisson Solve
    bV = SetbV_3D(p, epsilon);
    
    spy(AV)
    
    
%     newV = U\(L\bV);  %much faster to solve pre-factorized matrix. Not applicable to cont. eqn. b/c matrices keep changing.
    
    
%     dense_AV = full(AV);  % use this if not using compressed storage in C++
    
    % loop through the dense_AV matrix to store the information in
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
    
    values = zeros(N^3, 7);
    col_indices = zeros(N^3, 7);
    
    for row = 1:size(AV, 1)
        non_zero_col_cnt = 0;
        for col = 1:size(AV, 1)
            if AV(row, col) ~= 0 
                non_zero_col_cnt = non_zero_col_cnt + 1;
                values(row, non_zero_col_cnt) = AV(row, col);
                col_indices(row, non_zero_col_cnt) = col;
            end
        end
    end
                
    values;
    col_indices;
   

    tic
    
%     pcg(AV, bV, 1e-8);
    
    toc
    initial_guess = zeros(size(AV,1), 1);
    
    % NOTE: WE MUST TRANSPOSE THE MATRICES IN ORDER FOR THE OUTPUT THE HDF5
    % TO MATCH HOW READING IT IN C++!!!   
    
    h5create('cg_3D_poisson_sparse.h5','/matrix_values',size(values.'));
    h5create('cg_3D_poisson_sparse.h5','/col_indices',size(col_indices.'));
    h5create('cg_3D_poisson_sparse.h5','/solution',size(bV)); % prepare data set for solution
    h5create('cg_3D_poisson_sparse.h5','/initial_guess',size(bV));
    h5create('cg_3D_poisson_sparse.h5','/rhs',size(bV));
    h5create('cg_3D_poisson_sparse.h5','/cpu_time',1);  % spot to store the cpu time
    h5create('cg_3D_poisson_sparse.h5', '/cpu_per_iter', 1);
    h5create('cg_3D_poisson_sparse.h5','/num_iters',1); %store number of iters it took to solve
    h5create('cg_3D_poisson_sparse.h5','/tolerance',1);
    h5create('cg_3D_poisson_sparse.h5','/error',size(bV));
    
    h5write('cg_3D_poisson_sparse.h5', '/matrix_values', values.');
    h5write('cg_3D_poisson_sparse.h5', '/col_indices', col_indices.');
    h5write('cg_3D_poisson_sparse.h5', '/initial_guess',initial_guess);
    h5write('cg_3D_poisson_sparse.h5', '/rhs', bV);
    
end

size(AV)
size(values)

