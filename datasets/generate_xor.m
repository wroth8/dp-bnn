% Generates noisy XOR data for binary classification tasks.
%
% @author Wolfgang Roth

close all;
clear;

rng(1);

N = 5;
x = randn(N, 2) * 1e-1;
x = [x; bsxfun(@plus, randn(N, 2) * 1e-1, [1, 1])];
x = [x; bsxfun(@plus, randn(N, 2) * 1e-1, [0, 1])];
x = [x; bsxfun(@plus, randn(N, 2) * 1e-1, [1, 0])];
t = [zeros(2 * N, 1); ones(2 * N, 1)];

save('xor.mat', 'x', 't');

figure;
plot(x(t == 0, 1), x(t == 0, 2), 'bo'); hold on;
plot(x(t == 1, 1), x(t == 1, 2), 'rx');