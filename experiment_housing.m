% Performs a sampling experiment for the Boston housing dataset as it was
% done in the paper.
%
% @author Wolfgang Roth

close all;
clear;

addpath('code_matlab');
addpath('code_native');
addpath(genpath('code_ahmc'));

for fold_idx = 1:5
  % Perform 100 initial sampleZ runs
  input_file_data = 'datasets/housing.mat';
  input_file_fold = ['datasets/folds/housing_fold', num2str(fold_idx), '.mat'];
  output_file = ['experiming_housing_fold', num2str(fold_idx), '_sampleZinit.mat'];
  layout = [13,50,50,1];
  task = 'regress';
  sharing = 'layerwise';
  activation = 'tanh';
  alpha = 1e3;
  beta = 1e-2;
  gamma = 1e0;
  init_gamma = 1e0;
  seed = 1;
  batch_size = 0;
  discretization = 10;
  interpolation_method = 'pchip';
  n_iterations = 100;
  job_dpnnSampleZInterpolate_CV(input_file_data, input_file_fold, output_file, layout, task, sharing, activation, ...
      alpha, beta, gamma, init_gamma, seed, batch_size, discretization, interpolation_method, n_iterations);

  % Perform alternating z-sampling and w-sampling
  input_file_data = 'datasets/housing.mat';
  input_file_fold = ['datasets/folds/housing_fold', num2str(fold_idx), '.mat'];
  input_file_model = ['experiming_housing_fold', num2str(fold_idx), '_sampleZinit.mat'];
  output_file_prefix = ['experiment_housing_fold', num2str(fold_idx), '_sampleZ_ahmc'];
  layout = [13,50,50,1];
  task = 'regress';
  sharing = 'layerwise';
  activation = 'tanh';
  alpha = 1e3;
  beta = 1e-2;
  gamma = 1e0;
  seed = 1;
  batch_size = 0;
  discretization = 10;
  interpolation_method = 'pchip';
  LF_min_eps = 1e-6;
  LF_max_eps = 1e-1;
  LF_min_L = 1;
  LF_max_L = 250;
  ahmc_n_iterations = 400;
  ahmc_n_burnin = 200;
  samplezinit_n_iterations = 100;
  n_iterations = 24;
  job_dpnnSampleZInterpolateAhmcWithStart_samplezinit_CV(input_file_data, input_file_fold, input_file_model, ...
      output_file_prefix, layout, task, sharing, activation, alpha, beta, gamma, seed, batch_size, discretization, ...
      interpolation_method, LF_min_eps, LF_max_eps, LF_min_L, LF_max_L, ahmc_n_iterations, ahmc_n_burnin, ...
      samplezinit_n_iterations, n_iterations);
end

%% Evaluate the results
errs_tr_all = [];
errs_te_all = [];
lls_tr_all = [];
lls_te_all = [];
for fold_idx = 1:5
  result_file = ['experiment_housing_fold', num2str(fold_idx), '_sampleZ_ahmc_24_errors.mat'];
  load(result_file, 'errs_tr', 'errs_te', 'lls_tr', 'lls_te');
  
  errs_tr_all = [errs_tr_all; cat(2, errs_tr{:})];
  errs_te_all = [errs_te_all; cat(2, errs_te{:})];
  lls_tr_all = [lls_tr_all; cat(2, lls_tr{:})];
  lls_te_all = [lls_te_all; cat(2, lls_te{:})];
end

errs_tr_mean = mean(errs_tr_all, 1);
errs_te_mean = mean(errs_te_all, 1);
lls_tr_mean = mean(lls_tr_all, 1);
lls_te_mean = mean(lls_te_all, 1);
errs_tr_stderr = std(errs_tr_all, 0, 1) / sqrt(size(errs_tr_all, 1));
errs_te_stderr = std(errs_te_all, 0, 1) / sqrt(size(errs_tr_all, 1));
lls_tr_stderr = std(lls_tr_all, 0, 1) / sqrt(size(errs_tr_all, 1));
lls_te_stderr = std(lls_te_all, 0, 1) / sqrt(size(errs_tr_all, 1));

fprintf('Results for averaging %d samples:\n', length(errs_tr_mean));
fprintf('RMSE_TR: % 8f +- %8f\n', errs_tr_mean(end), errs_tr_stderr(end));
fprintf('RMSE_TE: % 8f +- %8f\n', errs_te_mean(end), errs_te_stderr(end));
fprintf('LL_TR:   % 8f +- %8f\n', lls_tr_mean(end), lls_tr_stderr(end));
fprintf('LL_TE:   % 8f +- %8f\n', lls_te_mean(end), lls_te_stderr(end));

figure;
hold on;
fill([1:length(errs_tr_stderr), length(errs_tr_stderr):-1:1], ...
     [errs_tr_mean-errs_tr_stderr, fliplr(errs_tr_mean+errs_tr_stderr)] ,[0.8, 0.8, 1], ...,
     'linestyle','none');
fill([1:length(errs_te_stderr), length(errs_te_stderr):-1:1], ...
     [errs_te_mean-errs_te_stderr, fliplr(errs_te_mean+errs_te_stderr)] ,[0.8, 1, 0.8], ...,
     'linestyle','none');
plot(1:length(errs_tr_mean), errs_tr_mean, 'b');
plot(1:length(errs_te_mean), errs_te_mean, 'g');

figure;
hold on;
fill([1:length(lls_tr_stderr), length(lls_tr_stderr):-1:1], ...
     [lls_tr_mean-lls_tr_stderr, fliplr(lls_tr_mean+lls_tr_stderr)] ,[0.8, 0.8, 1], ...,
     'linestyle','none');
fill([1:length(lls_te_stderr), length(lls_te_stderr):-1:1], ...
     [lls_te_mean-lls_te_stderr, fliplr(lls_te_mean+lls_te_stderr)] ,[0.8, 1, 0.8], ...,
     'linestyle','none');
plot(1:length(lls_tr_mean), lls_tr_mean, 'b');
plot(1:length(lls_te_mean), lls_te_mean, 'g');
