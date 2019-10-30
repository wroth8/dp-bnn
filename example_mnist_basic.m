% Demonstrates DP neural network on MNIST Basic.
%
% @author Wolfgang Roth

close all;
clear;

addpath('code_matlab');
addpath('code_native');

load('datasets/mnist_basic_pca50_zeromean.mat');

dpnn_layout = [50,50,50,10];
dpnn_task = 'muclass';
dpnn_sharing = 'layerwise';
dpnn_activation = 'tanh';
dpnn_alpha = 10; % DP concentration parameter
dpnn_beta = nan; % beta is only relevant for regression
dpnn_gamma = 1e0; % variance of the weight prior
dpnn_init_gamma = 1e0; % variance used to initialize the weights using a Gaussian
dpnn_rng_seed = 1;
model = dpnnInitCRP(dpnn_layout, dpnn_task, dpnn_sharing, dpnn_activation, ...
    dpnn_alpha, dpnn_beta, dpnn_gamma, dpnn_init_gamma, dpnn_rng_seed);

%% Train a BFGS model and evaluate its performance.
% Note that the weights are shared randomly according to a Chinese
% restaurant process.
[model_bfgs, bfgs_err, bfgs_perf] = dpnnTrainBFGS(model, {x_tr, x_va, x_te}, {t_tr, t_va, t_te}, 1000);

ce_bfgs_tr = dpnnEvaluate(model_bfgs, x_tr, t_tr);
ce_bfgs_va = dpnnEvaluate(model_bfgs, x_va, t_va);
ce_bfgs_te = dpnnEvaluate(model_bfgs, x_te, t_te);
fprintf('DPNN BFGS: CE_TR=%6.3f, CE_VA=%6.3f, CE_TE=%6.3f\n', ce_bfgs_tr, ce_bfgs_va, ce_bfgs_te);

%% Do five iterations of z-sampling
% This should improve the sharing over the random sharing according to the
% Chinese restaurant process.
num_iterations_sampleZ = 5;

% SampleZ parameters
batch_size = 0;          % Disable batch_size
approx_N = 10;           % Disable approximation
m = 100;                 % #auxiliary variables
verbosity = 2;           % Print output at neuron level
approx_method = 'pchip'; % Interpolation method (one of 'none', 'nearest', 'linear', 'pchip')

model = model_bfgs;
models_sampleZ = cell(1, num_iterations_sampleZ);
ce_sampleZ_tr = zeros(1, num_iterations_sampleZ);
ce_sampleZ_va = zeros(1, num_iterations_sampleZ);
ce_sampleZ_te = zeros(1, num_iterations_sampleZ);
for ii = 1:num_iterations_sampleZ
  fprintf('sampleZ iteration %d/%d\n', ii, num_iterations_sampleZ);
  tic;
  models_sampleZ{ii} = mexSampleZInterpolate(model, x_tr, t_tr, batch_size, approx_N, m, verbosity, approx_method);
  t_elapsed = toc;

  ce_sampleZ_tr(ii) = dpnnEvaluate(models_sampleZ{ii}, x_tr, t_tr);
  ce_sampleZ_va(ii) = dpnnEvaluate(models_sampleZ{ii}, x_va, t_va);
  ce_sampleZ_te(ii) = dpnnEvaluate(models_sampleZ{ii}, x_te, t_te);
  fprintf('DPNN sampleZ: CE_TR=%6.3f, CE_VA=%6.3f, CE_TE=%6.3f (t_elapsed[sampleZ]=%f seconds)\n', ...
      ce_sampleZ_tr(ii), ce_sampleZ_va(ii), ce_sampleZ_te(ii), t_elapsed);
end

figure;
plot(1:(length(ce_sampleZ_tr)+1), [ce_bfgs_tr, ce_sampleZ_tr], 'b'); hold on;
plot(1:(length(ce_sampleZ_va)+1), [ce_bfgs_va, ce_sampleZ_va], 'r');
plot(1:(length(ce_sampleZ_te)+1), [ce_bfgs_te, ce_sampleZ_te], 'g');
ylim([0, 0.2]);

%% Obtain 1000 samples with w-sampling starting from the model obtained with z-sampling

sghmc_batch_size = 100;
sghmc_num_samples_total = 1200;
sghmc_num_samples_burnin = 200;
sghmc_step_size = 3e-9;
sghmc_alpha = 5e-2;
sghmc_do_output = true;
tic;
[ sghmc_out, model_sghmc ] = dpnnSampleWeightsSGHMC(models_sampleZ{end}, x_tr, t_tr, sghmc_batch_size, sghmc_num_samples_total, ...
    sghmc_num_samples_burnin, sghmc_step_size, sghmc_alpha, sghmc_do_output);
t_elapsed = toc;

% Stack all weight samples contained in sghmc_out.saved into model_sghmc
if strcmp(model_sghmc.sharing, 'layerwise')
  start_idx = 1;
  for l = 1:model_sghmc.num_layers
    end_idx = start_idx + model_sghmc.num_unique_weights(l) - 1;
    model_sghmc.W{l} = sghmc_out.saved(:, start_idx:end_idx);
    start_idx = end_idx + 1;
  end
else
  model_sghmc.W = sghmc_out.saved;
end

% Evaluate the ensemble
ce_sghmc_tr = dpnnEvaluateEnsembleForward(model_sghmc, x_tr, t_tr);
ce_sghmc_va = dpnnEvaluateEnsembleForward(model_sghmc, x_va, t_va);
ce_sghmc_te = dpnnEvaluateEnsembleForward(model_sghmc, x_te, t_te);
fprintf('DPNN SGHMC: CE_TR=%6.3f, CE_VA=%6.3f, CE_TE=%6.3f (t_elapsed[sampleZ]=%f seconds)\n', ...
      ce_sghmc_tr{1}(end), ce_sghmc_va{1}(end), ce_sghmc_te{1}(end), t_elapsed);

figure;
plot(1:length(ce_sghmc_tr{1}), ce_sghmc_tr{1}, 'b'); hold on;
plot(1:length(ce_sghmc_va{1}), ce_sghmc_va{1}, 'r');
plot(1:length(ce_sghmc_te{1}), ce_sghmc_te{1}, 'g');

%% Alternately do z-sampling and w-sampling (SGHMC)
% We start again from the BFGS model and alternately do z-sampling and
% weight sampling. We can see that the performance improves considerably
% showing that we can benefit from the different sharing structures
% obtained via z-sharing (note that in the above w-sampling experiment the
% sharing is always the same).

num_iterations = 5;

% sampleZ parameters
batch_size = 0;          % Disable batch_size
approx_N = 10;           % Disable approximation
m = 100;                 % #auxiliary variables
verbosity = 2;           % Print output at neuron level
approx_method = 'pchip'; % Interpolation method (one of 'none', 'nearest', 'linear', 'pchip')

% SGHMC parameters
sghmc_batch_size = 100;
sghmc_num_samples_total = 240;
sghmc_num_samples_burnin = 40;
sghmc_step_size = 3e-9;
sghmc_alpha = 5e-2;
sghmc_do_output = true;

model = model_bfgs;
model_ensemble = [];
for ii = 1:num_iterations
  fprintf('sampleZ iteration %d/%d\n', ii, num_iterations);
  tic;
  model = mexSampleZInterpolate(model, x_tr, t_tr, batch_size, approx_N, m, verbosity, approx_method);
  t_elapsed_sampleZ = toc;
  
  fprintf('sghmc iteration %d/%d\n', ii, num_iterations);
  tic;
  [ sghmc_out, model ] = dpnnSampleWeightsSGHMC(model, x_tr, t_tr, sghmc_batch_size, ...
      sghmc_num_samples_total, sghmc_num_samples_burnin, sghmc_step_size, sghmc_alpha, sghmc_do_output);
  t_elapsed_sghmc = toc;

  % Stack all weight samples contained in sghmc_out.saved into model_sghmc
  model_sghmc = model;
  if strcmp(model_sghmc.sharing, 'layerwise')
    start_idx = 1;
    for l = 1:model_sghmc.num_layers
      end_idx = start_idx + model_sghmc.num_unique_weights(l) - 1;
      model_sghmc.W{l} = sghmc_out.saved(:, start_idx:end_idx);
      start_idx = end_idx + 1;
    end
  else
    model_sghmc.W = sghmc_out.saved;
  end

  fprintf('Iteration %d/%d finished in %f seconds [t_sampleZ=%f seconds, t_sghmc=%f seconds]\n', ii, num_iterations, ...
      t_elapsed_sampleZ + t_elapsed_sghmc, t_elapsed_sampleZ);
  
  model_ensemble = [model_ensemble, model_sghmc];
end

ce_sghmc_tr = dpnnEvaluateEnsembleForward(model_ensemble, x_tr, t_tr);
ce_sghmc_va = dpnnEvaluateEnsembleForward(model_ensemble, x_va, t_va);
ce_sghmc_te = dpnnEvaluateEnsembleForward(model_ensemble, x_te, t_te);
ce_sghmc_tr = cat(2, ce_sghmc_tr{:});
ce_sghmc_va = cat(2, ce_sghmc_va{:});
ce_sghmc_te = cat(2, ce_sghmc_te{:});
fprintf('DPNN SGHMC: CE_TR=%6.3f, CE_VA=%6.3f, CE_TE=%6.3f (t_elapsed[sampleZ]=%f seconds)\n', ...
      ce_sghmc_tr(end), ce_sghmc_va(end), ce_sghmc_te(end), t_elapsed);

figure;
plot(1:length(ce_sghmc_tr), ce_sghmc_tr, 'b'); hold on;
plot(1:length(ce_sghmc_va), ce_sghmc_va, 'r');
plot(1:length(ce_sghmc_te), ce_sghmc_te, 'g');
