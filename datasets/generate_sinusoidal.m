% Generates noisy data from a sinusoidal function in (0,1) for regression
% tasks.
%
% @author Wolfgang Roth

close all;
clear;

rng(1);

N = 20;
x = rand(N, 1);
t = sin(x * 2 * pi) + randn(N, 1) * 1e-1;

save('sinusoidal.mat', 'x', 't');

figure;
plot(x, t, 'bo');