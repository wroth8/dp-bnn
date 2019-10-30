% Demonstrates DP neural network on a simple multiclass classification task
% with sinusoidal data.
%
% @author Wolfgang Roth

close all;
clear;

addpath('code_matlab');
addpath('code_native');

load('datasets/multiclass.mat');

dpnn_layout = [2,25,25,4];
dpnn_task = 'muclass';
dpnn_sharing = 'layerwise';
dpnn_activation = 'relu';
dpnn_alpha = 1; % DP concentration parameter
dpnn_beta = nan; % beta is only relevant for regression
dpnn_gamma = 1; % variance of the weight prior
dpnn_init_gamma = 1; % variance used to initialize the weights using a Gaussian
dpnn_rng_seed = 2;
model = dpnnInitCRP(dpnn_layout, dpnn_task, dpnn_sharing, dpnn_activation, ...
    dpnn_alpha, dpnn_beta, dpnn_gamma, dpnn_init_gamma, dpnn_rng_seed);

%% Alternately perform z-sampling and w-sampling and average results
x_vals = linspace(-5, 10, 150);
y_vals = linspace(-5, 10, 150);
[X,Y] = meshgrid(x_vals, y_vals);
[~, Z_max] = max(dpnnFunction(model, [X(:), Y(:)]), [], 2);
Z = reshape(Z_max, [length(x_vals), length(y_vals)]);
figure(1);
clf;
contourf(X, Y, Z); hold on;
plot(x(t == 1, 1), x(t == 1, 2), 'bo');
plot(x(t == 2, 1), x(t == 2, 2), 'rx');
plot(x(t == 3, 1), x(t == 3, 2), 'gd');
plot(x(t == 4, 1), x(t == 4, 2), 'k*');
xlim([-5,10]);
ylim([-5,10]);
caxis([1, 4]);
drawnow;

num_samples = 50;
plot_z_avg = zeros(length(x_vals), length(y_vals), 4, num_samples);
Z_avg = [];

for it = 1:num_samples
  % SampleZ parameters
  batch_size = 0;          % Disable batch_size
  approx_N = 10;           % Disable approximation
  m = 100;                 % #auxiliary variables
  verbosity = 0;           % Disable output to stdout
  approx_method = 'pchip'; % Interpolation method (one of 'none', 'nearest', 'linear', 'pchip')

  % Hybrid Monte Carlo parameters
  num_hmc_iter = 3;        % number of hybrid Monte Carlo iterations
  leapfrog_L = 100;        % number of leapfrog iterations
  leapfrog_epsilon = 1e-3; % leapfrog step size

  model = mexSampleZInterpolate(model, x, t, batch_size, approx_N, m, verbosity, approx_method);
  [out, model] = dpnnSampleWeightsHMC(model, x, t, num_hmc_iter, leapfrog_L, leapfrog_epsilon, true);
  fprintf('#unique weights: %d (#total weights: %d)\n', sum(model.num_unique_weights), sum((model.layout(1:end-1) + 1).*model.layout(2:end)));

  Z = reshape(dpnnFunction(model, [X(:), Y(:)]), [length(x_vals), length(y_vals), 4]);
  plot_z_avg(:, :, :, it) = Z;
  [~, Z] = max(Z, [], 3);

  % Plot prediction sample
  figure(1);
  clf;
  contourf(X, Y, Z); hold on;
  plot(x(t == 1, 1), x(t == 1, 2), 'bo');
  plot(x(t == 2, 1), x(t == 2, 2), 'rx');
  plot(x(t == 3, 1), x(t == 3, 2), 'gd');
  plot(x(t == 4, 1), x(t == 4, 2), 'k*');
  xlim([-5,10]);
  ylim([-5,10]);
  caxis([1, 4]);
  title(sprintf('Iteration %d', it));
  drawnow;

  % Plot averaged prediction over all samples
  [~, Z] = max(mean(plot_z_avg(:,:,:,1:it), 4), [], 3);
  figure(2);
  clf;
  contourf(X, Y, Z); hold on;
  plot(x(t == 1, 1), x(t == 1, 2), 'bo');
  plot(x(t == 2, 1), x(t == 2, 2), 'rx');
  plot(x(t == 3, 1), x(t == 3, 2), 'gd');
  plot(x(t == 4, 1), x(t == 4, 2), 'k*');
  xlim([-5,10]);
  ylim([-5,10]);
  caxis([1, 4]);
  title(sprintf('Iteration %d', it));
  drawnow;
end

%% Train a model with BFGS and plot the learned function
% If BFGS training does not result in reasonable predictions, consider
% running the sampling algorithm above first to obtain a better sharing.
model = dpnnTrainBFGS(model, x, t, 500, true);
x_vals = linspace(-5, 10, 150);
y_vals = linspace(-5, 10, 150);
[X,Y] = meshgrid(x_vals, y_vals);
[~, Z_max] = max(dpnnFunction(model, [X(:), Y(:)]), [], 2);
Z = reshape(Z_max, [length(x_vals), length(y_vals)]);
figure(1);
clf;
contourf(X, Y, Z); hold on;
plot(x(t == 1, 1), x(t == 1, 2), 'bo');
plot(x(t == 2, 1), x(t == 2, 2), 'rx');
plot(x(t == 3, 1), x(t == 3, 2), 'gd');
plot(x(t == 4, 1), x(t == 4, 2), 'k*');
xlim([-5,10]);
ylim([-5,10]);
caxis([1, 4]);
drawnow;

%% Train a model with ADAM and plot the learned function
% If ADAM training does not result in reasonable predictions, consider
% running the sampling algorithm above first to obtain a better sharing.
[model, err, perf] = dpnnTrainADAM(model, x, t, 10, 3e-3, 5000, true);
x_vals = linspace(-5, 10, 150);
y_vals = linspace(-5, 10, 150);
[X,Y] = meshgrid(x_vals, y_vals);
[~, Z_max] = max(dpnnFunction(model, [X(:), Y(:)]), [], 2);
Z = reshape(Z_max, [length(x_vals), length(y_vals)]);
figure(1);
clf;
contourf(X, Y, Z); hold on;
plot(x(t == 1, 1), x(t == 1, 2), 'bo');
plot(x(t == 2, 1), x(t == 2, 2), 'rx');
plot(x(t == 3, 1), x(t == 3, 2), 'gd');
plot(x(t == 4, 1), x(t == 4, 2), 'k*');
xlim([-5,10]);
ylim([-5,10]);
caxis([1, 4]);
drawnow;