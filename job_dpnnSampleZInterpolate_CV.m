function job_dpnnSampleZInterpolate_CV( input_file_data, input_file_fold, output_file, layout, task, sharing, activation, alpha, beta, gamma, init_gamma, seed, batch_size, discretization, interpolation_method, n_iterations )
% job-script for performing several runs of mexSampleZInterpolate with
% cross validation folds.
%
% @author Wolfgang Roth

fprintf('---------------------------------------------------------------------------\n');
fprintf('job_dpnnSampleZInterpolate_CV\n');
fprintf('input_file_data: %s\n', input_file_data);
fprintf('input_file_fold: %s\n', input_file_fold);
fprintf('output_file: %s\n', output_file);
fprintf(['layout: [ ', repmat('%d ', size(layout)), ']\n'], layout);
fprintf('task: %s\n', task);
fprintf('sharing: %s\n', sharing);
fprintf('activation: %s\n', activation);
fprintf('alpha: %f\n', alpha);
fprintf('beta: %f\n', beta);
fprintf('gamma: %f\n', gamma);
fprintf('init_gamma: %f\n', init_gamma);
fprintf('seed: %d\n', seed);
fprintf('batch_size: %d\n', batch_size);
fprintf('discretization: %d\n', discretization);
fprintf('interpolation_method: %s\n', interpolation_method);
fprintf('n_iterations: %d\n', n_iterations);
fprintf('---------------------------------------------------------------------------\n');

load(input_file_data, 'x', 't');
load(input_file_fold, 'idx_tr', 'idx_te');

% Preprocess the data.
% Normalize the training data to have zero mean and unit variance
if ~isempty(intersect(idx_tr, idx_te))
  error('Error in ''job_dpnnSampleZInterpolate_CV'': ''idx_tr'' and ''idx_te'' must not have same indices');  
end
if length(intersect(union(idx_tr, idx_te), 1:size(x, 1))) ~= size(x, 1)
  error('Error in ''job_dpnnSampleZInterpolate_CV'': ''idx_tr'' and ''idx_te'' do not contain all indices');   
end

x_tr = x(idx_tr, :);
t_tr = t(idx_tr, :);

x_mean = mean(x_tr, 1);
x_std = std(x_tr, 1);
x_std(x_std == 0) = 1;
x_tr = bsxfun(@times, bsxfun(@minus, x_tr, x_mean), 1 ./ x_std);

% In case of regression also normalize the targets.
if strcmp(task, 'regress')
  t_mean = mean(t_tr, 1);
  t_std = std(t_tr, 1);
  t_std(t_std == 0) = 1;
  t_tr = bsxfun(@times, bsxfun(@minus, t_tr, t_mean), 1 ./ t_std);
end

if exist(output_file, 'file') ~= 2
  fprintf('No result files for this job available - creating new DPNN\n');
  model = dpnnInitCRP(layout, task, sharing, activation, alpha, beta, gamma, init_gamma, seed);
  model % Display information about the model
  result = [];
else
  % The file already exists and does not have to be computed
  fprintf('File ''%s'' already available\n', output_file);
  load(output_file, 'result');
  rng(result(end).rng_matlab);
  mexRngInit(result(end).rng_mex);
  model = result(end).model;
end

for ii = (length(result) + 1):n_iterations
  fprintf('Iteration %d/%d\n', ii, n_iterations);
  tic;
  model = mexSampleZInterpolate(model, x_tr, t_tr, batch_size, discretization, 100, 2, interpolation_method);
  time_needed = toc;
  fprintf('Time needed: %f seconds\n', time_needed);
  rng_matlab = rng;
  rng_mex = mexRngState;
  result = [result, struct('model', model, 'time_needed', time_needed, 'rng_matlab', rng_matlab, 'rng_mex', rng_mex)];
  save(output_file, 'result');
end

fprintf('job_dpnnSampleZInterpolate_CV finished\n');

end

