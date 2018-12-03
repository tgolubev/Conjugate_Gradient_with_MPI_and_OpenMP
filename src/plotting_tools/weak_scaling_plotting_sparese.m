% Plotting smooth lines with markers

% to do this need to use cubic spline interpolant

%NOTE: to make the graph edges thicker, click on the plot region edge, go
%into more properties and change line width to i.e. 1.0

%TO USE: just need to manually enter the values of proc1 proc2 proc4 and proc8
%timings etc...
% as arrays: i.e. threads1 = [1	1.406803093	1.911696753	2.231995189 2.335761356]

% this is for scaling studies
x1 = [1 2 4 8];


plot(x1, threads1);
hold on
plot(x1, threads2);
plot(x1, threads4);

scatter(x1, threads1, 's', 'filled', 'black')  %'-o' tells to add markers, filled to fill them in
axis([0 9 0 inf])
hold on

scatter(x1, threads2, 's', 'filled', 'black')
scatter(x1, threads4, 's', 'filled', 'black')

xlabel("MPI Ranks", 'FontSize', 24)
ylabel("CPU Time per CG Iter (s)", 'FontSize', 24)
title("Weak Scaling", 'FontSize', 28)
legend({'1 thread', '2 threads', '4 threads'});
ax = gca;
ax.FontSize = 24;
grid on



