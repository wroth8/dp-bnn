% Generates data from four different Gaussians for multiclass
% classification.
%
% @author Wolfgang Roth

close all;
clear;

rng(1);

mu1 = [-3, -3];
Sigma1 = [1 , 0; 0 , 1];

mu2 = [5, 5];
Sigma2 = 2 * [1; 1] * [1, 1] + [1; -1] * [1, -1];

mu3 = [-1, 3];
Sigma3 = 3 * [1; -1] * [1, -1] + [1; 1] * [1, 1];

mu4 = [5, -2];
Sigma4 = 2 * [1; 1] * [1, 1] + [1; -1] * [1, -1];

N = 10;

x = mvnrnd(mu1, Sigma1, N);
t = ones(N, 1);

x = [x; mvnrnd(mu2, Sigma2, N)];
t = [t; ones(N, 1) * 2];

x = [x; mvnrnd(mu3, Sigma3, N)];
t = [t; ones(N, 1) * 3];

x = [x; mvnrnd(mu4, Sigma4, N)];
t = [t; ones(N, 1) * 4];

save('multiclass.mat', 'x', 't');

figure;
plot(x(t == 1, 1), x(t == 1, 2), 'bo'); hold on;
plot(x(t == 2, 1), x(t == 2, 2), 'rx');
plot(x(t == 3, 1), x(t == 3, 2), 'gd');
plot(x(t == 4, 1), x(t == 4, 2), 'k*');