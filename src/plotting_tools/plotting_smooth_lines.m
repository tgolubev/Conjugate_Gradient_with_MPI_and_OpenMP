% Plotting smooth lines with markers

% to do this need to use cubic spline interpolant

%NOTE: to make the graph edges thicker, click on the plot region edge, go
%into more properties and change line width to i.e. 1.0

%TO USE: just need to manually enter the values of proc1 proc2 proc4 and proc8
%timings etc...
% as arrays: i.e. proc1 = [1	1.406803093	1.911696753	2.231995189 2.335761356]

% this is for scaling studies
x1 = [1 2 4 8 10];
x2 = x1(1:3);           %sometimes just have the 1 2 4 was run


cs1 = csapi(x1, proc1);
cs2 = csapi(x1, proc2);
cs3 = csapi(x1, proc4);
cs4 = csapi(x2, proc8);

fnplt(cs1, 2)
hold on
fnplt(cs2, 2)
fnplt(cs3, 2)
fnplt(cs4, 2)

scatter(x1, proc1, 's', 'filled', 'black')  %'-o' tells to add markers, filled to fill them in
axis([0 11 0 4])
hold on

scatter(x1, proc2, 's', 'filled', 'black')
scatter(x1, proc4, 's', 'filled', 'black')
scatter(x2, proc8, 's', 'filled', 'black')

xlabel("Number of OpenMP Threads", 'FontSize', 24)
ylabel("Relative Speedup", 'FontSize', 24)
title("Strong Scaling", 'FontSize', 28)
legend({'1 rank', '2 ranks', '4 ranks', '8 ranks'});
ax = gca;
ax.FontSize = 24
grid on



