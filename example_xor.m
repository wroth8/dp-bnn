% Demonstrates DP neural network on a simple binary classification task
% with xor data.
%
% @author Wolfgang Roth

close all;
clear;

addpath('code_matlab');
addpath('code_native');

load('datasets/xor.mat');

dpnn_layout = [2,50,50,1];
dpnn_task = 'biclass';
dpnn_sharing = 'layerwise';
dpnn_activation = 'tanh';
dpnn_alpha = 1; % DP concentration parameter
dpnn_beta = nan; % beta is only relevant for regression
dpnn_gamma = 1; % variance of the weight prior
dpnn_init_gamma = 1; % variance used to initialize the weights using a Gaussian
dpnn_rng_seed = 2;
model = dpnnInitCRP(dpnn_layout, dpnn_task, dpnn_sharing, dpnn_activation, ...
    dpnn_alpha, dpnn_beta, dpnn_gamma, dpnn_init_gamma, dpnn_rng_seed);

%% Alternately perform z-sampling and w-sampling and average results
x_vals = linspace(-0.5, 1.5, 20);
y_vals = linspace(-0.5, 1.5, 20);
[X,Y] = meshgrid(x_vals, y_vals);
Z = reshape(dpnnFunction(model, [X(:), Y(:)]), size(X));
figure(1);
clf;
contourf(X, Y, Z); hold on;
contour(X, Y, Z, 0.5, 'Color', [0 0 0], 'LineWidth', 2);
plot(x(t == 0, 1), x(t == 0, 2), 'go'); hold on;
plot(x(t == 1, 1), x(t == 1, 2), 'gx');
caxis([0, 1]);
drawnow;

num_samples = 100;
plot_z_avg = zeros(length(x_vals), length(y_vals), num_samples);

for it = 1:100
  % SampleZ parameters
  batch_size = 0;          % Disable batch_size
  approx_N = 10;           % How many supporting points for interpolation (0 == disable interpolation)
  m = 100;                 % #auxiliary variables for Neal Algorithm 8
  verbosity = 0;           % Disable output to stdout
  approx_method = 'pchip'; % Interpolation method (one of 'none', 'nearest', 'linear', 'pchip')

  % Hybrid Monte Carlo parameters
  num_hmc_iter = 3;        % number of hybrid Monte Carlo iterations
  leapfrog_L = 100;        % number of leapfrog iterations
  leapfrog_epsilon = 1e-1; % leapfrog step size

  model = mexSampleZInterpolate(model, x, t, batch_size, approx_N, m, verbosity, approx_method);
  [out, model] = dpnnSampleWeightsHMC(model, x, t, num_hmc_iter, leapfrog_L, leapfrog_epsilon, true);
  fprintf('#unique weights: %d (#total weights: %d)\n', sum(model.num_unique_weights), sum((model.layout(1:end-1) + 1).*model.layout(2:end)));

  % Plot prediction sample
  Z = reshape(dpnnFunction(model, [X(:), Y(:)]), size(X));
  figure(1);
  clf;
  contourf(X, Y, Z); hold on;
  contour(X, Y, Z, 0.5, 'Color', [0 0 0], 'LineWidth', 2);
  plot(x(t == 0, 1), x(t == 0, 2), 'go'); hold on;
  plot(x(t == 1, 1), x(t == 1, 2), 'gx');
  caxis([0, 1]);
  title(sprintf('Iteration %d', it));
  drawnow;

  % Plot averaged prediction over all samples
  plot_z_avg(:,:,it) = Z;
  figure(2);
  clf;
  contourf(X, Y, mean(plot_z_avg(:,:,1:it),3)); hold on;
  contour(X, Y, mean(plot_z_avg(:,:,1:it),3), 0.5, 'Color', [0 0 0], 'LineWidth', 2);
  plot(x(t == 0, 1), x(t == 0, 2), 'go'); hold on;
  plot(x(t == 1, 1), x(t == 1, 2), 'gx');
  caxis([0, 1]);
  title(sprintf('Iteration %d', it));
  drawnow;
end


%% Train a model with BFGS and plot the learned function
% If BFGS training does not result in reasonable predictions, consider
% running the sampling algorithm above first to obtain a better sharing.
model = dpnnTrainBFGS(model, x, t, 500, true);
x_vals = linspace(-0.5, 1.5, 20);
y_vals = linspace(-0.5, 1.5, 20);
[X,Y] = meshgrid(x_vals, y_vals);
Z = reshape(dpnnFunction(model, [X(:), Y(:)]), size(X));
figure(1);
clf;
contourf(X, Y, Z); hold on;
contour(X, Y, Z, 0.5, 'Color', [0 0 0], 'LineWidth', 2);
plot(x(t == 0, 1), x(t == 0, 2), 'go'); hold on;
plot(x(t == 1, 1), x(t == 1, 2), 'gx');
caxis([0, 1]);
drawnow;

%% Train a model with ADAM and plot the learned function
% If ADAM training does not result in reasonable predictions, consider
% running the sampling algorithm above first to obtain a better sharing.
[model, err, perf] = dpnnTrainADAM(model, x, t, 5, 3e-3, 5000, true);
x_vals = linspace(-0.5, 1.5, 20);
y_vals = linspace(-0.5, 1.5, 20);
[X,Y] = meshgrid(x_vals, y_vals);
Z = reshape(dpnnFunction(model, [X(:), Y(:)]), size(X));
figure(1);
clf;
contourf(X, Y, Z); hold on;
contour(X, Y, Z, 0.5, 'Color', [0 0 0], 'LineWidth', 2);
plot(x(t == 0, 1), x(t == 0, 2), 'go'); hold on;
plot(x(t == 1, 1), x(t == 1, 2), 'gx');
caxis([0, 1]);
drawnow;
