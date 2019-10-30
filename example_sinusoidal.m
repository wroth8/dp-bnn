% Demonstrates DP neural network on a simple regression task with
% sinusoidal data.
%
% @author Wolfgang Roth

close all;
clear;

addpath('code_matlab');
addpath('code_native');

load('datasets/sinusoidal.mat');

dpnn_layout = [1,50,50,1];
dpnn_task = 'regress';
dpnn_sharing = 'layerwise';
dpnn_activation = 'tanh';
dpnn_alpha = 1; % DP concentration parameter
dpnn_beta = 1e-1; % variance of the predicted outputs
dpnn_gamma = 1; % variance of the weight prior
dpnn_init_gamma = 1; % variance used to initialize the weights using a Gaussian
dpnn_rng_seed = 2;
model = dpnnInitCRP(dpnn_layout, dpnn_task, dpnn_sharing, dpnn_activation, ...
    dpnn_alpha, dpnn_beta, dpnn_gamma, dpnn_init_gamma, dpnn_rng_seed);

%% Alternately perform z-sampling and w-sampling and average results
figure(1);
clf;
plot_x = linspace(0, 1, 100);
plot_y = dpnnFunction(model, plot_x')';
plot(plot_x, plot_y); hold on;
plot(x, t, 'ro');
ylim([-1.5, 1.5]);
title(sprintf('Iteration %d', 0));
drawnow;

num_samples = 30;
plot_y_avg = zeros(length(plot_x), num_samples);

for it = 1:num_samples
  % SampleZ parameters
  batch_size = 0;          % Disable batch_size
  approx_N = 10;           % How many supporting points for interpolation (0 == disable interpolation)
  m = 100;                 % #auxiliary variables for Neal Algorithm 8
  verbosity = 0;           % Disable output to stdout
  approx_method = 'pchip'; % Interpolation method (one of 'none', 'nearest', 'linear', 'pchip')

  % Hybrid Monte Carlo parameters
  num_hmc_iter = 3;        % number of hybrid Monte Carlo iterations
  leapfrog_L = 100;        % number of leapfrog iterations
  leapfrog_epsilon = 2e-3; % leapfrog step size

  model = mexSampleZInterpolate(model, x, t, batch_size, approx_N, m, verbosity, approx_method);
  [out, model] = dpnnSampleWeightsHMC(model, x, t, num_hmc_iter, leapfrog_L, leapfrog_epsilon, true);
  fprintf('#unique weights: %d (#total weights: %d)\n', sum(model.num_unique_weights), sum((model.layout(1:end-1) + 1).*model.layout(2:end)));

  plot_y = dpnnFunction(model, plot_x')';
  plot_y_avg(:, it) = plot_y;
  figure(1);
  clf;
  plot(plot_x, plot_y, 'b'); hold on;
  plot(plot_x, mean(plot_y_avg(:, 1:it), 2), 'g');
  plot(x, t, 'ro');
  ylim([-1.5, 1.5]);
  title(sprintf('Iteration %d', it));
  drawnow;
end

%% Train a model with BFGS and plot the learned function
model = dpnnTrainBFGS(model, x, t, 500, true);
figure(1);
clf;
plot_x = linspace(0, 1, 100);
plot_y = dpnnFunction(model, plot_x')';
plot(plot_x, plot_y, 'b'); hold on;
plot(x, t, 'ro');
ylim([-1.5, 1.5]);
title('BFGS');

%% Train a model with ADAM and plot the learned function
[model, err, perf] = dpnnTrainADAM(model, x, t, 5, 3e-3, 5000, true);
figure(1);
clf;
plot_x = linspace(0, 1, 100);
plot_y = dpnnFunction(model, plot_x')';
plot(plot_x, plot_y, 'b'); hold on;
plot(x, t, 'ro');
ylim([-1.5, 1.5]);
title('ADAM');