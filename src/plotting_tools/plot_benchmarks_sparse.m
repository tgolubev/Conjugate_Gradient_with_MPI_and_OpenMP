% Plotting smooth lines with markers

% to do this need to use cubic spline interpolant

%NOTE: to make the graph edges thicker, click on the plot region edge, go
%into more properties and change line width to i.e. 1.0

%TO USE: just need to manually enter the values of proc1 proc2 proc4 and proc8
%timings etc...
% as arrays: i.e. matlab = [1	1.406803093	1.911696753	2.231995189 2.335761356]

% this is for scaling studies
x1 = [1 2 4 8 10];


cs1 = csapi(x1, matlab);
cs2 = csapi(x1, procs1);
cs3 = csapi(x1, procs4);

fnplt(cs1, 2)
hold on
fnplt(cs2, 2)
fnplt(cs3, 2)

scatter(x1, matlab, 's', 'filled', 'black')  %'-o' tells to add markers, filled to fill them in
axis([0 11 0 inf])
hold on

scatter(x1, procs1, 's', 'filled', 'black')
scatter(x1, procs4, 's', 'filled', 'black')

xlabel("Number of Threads", 'FontSize', 33)
ylabel("CPU Time per CG Iter (s)", 'FontSize', 33)
title("Sparse Matrix", 'FontSize', 33)
legend({'Matlab', 'C++ w/ 1 MPI Rank', 'C++ w/ 4 MPI Ranks'});
ax = gca;
ax.FontSize = 30;
grid on

% set(gca, 'YScale', 'log')



