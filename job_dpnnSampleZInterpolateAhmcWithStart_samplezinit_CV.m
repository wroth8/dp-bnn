function job_dpnnSampleZInterpolateAhmcWithStart_samplezinit_CV( input_file_data, input_file_fold, input_file_model, output_file_prefix, layout, task, sharing, activation, alpha, beta, gamma, seed, batch_size, discretization, interpolation_method, LF_min_eps, LF_max_eps, LF_min_L, LF_max_L, ahmc_n_iterations, ahmc_n_burnin, samplezinit_n_iterations, n_iterations )
% job-script for alternate sampling between mexSampleZInterpolate and ahmc.
% sampleZ is iterated to find a good structure to start with. Ahmc does not
% use optimization for its initial weights. Cross-validation is used.
%
% @author Wolfgang Roth

fprintf('---------------------------------------------------------------------------\n');
fprintf('job_dpnnSampleZInterpolateAhmcWithStart_samplezinit_CV\n');
fprintf('input_file_data: %s\n', input_file_data);
fprintf('input_file_fold: %s\n', input_file_fold);
fprintf('input_file_model: %s\n', input_file_model);
fprintf('output_file_prefix: %s\n', output_file_prefix);
fprintf(['layout: [ ', repmat('%d ', size(layout)), ']\n'], layout);
fprintf('task: %s\n', task);
fprintf('sharing: %s\n', sharing);
fprintf('activation: %s\n', activation);
fprintf('alpha: %f\n', alpha);
fprintf('beta: %f\n', beta);
fprintf('gamma: %f\n', gamma);
fprintf('seed: %d\n', seed);
fprintf('batch_size: %d\n', batch_size);
fprintf('discretization: %d\n', discretization);
fprintf('interpolation_method: %s\n', interpolation_method);
fprintf('LF_min_eps: %f\n', LF_min_eps);
fprintf('LF_max_eps: %f\n', LF_max_eps);
fprintf('LF_min_L: %d\n', LF_min_L);
fprintf('LF_max_L: %d\n', LF_max_L);
fprintf('ahmc_n_iterations: %d\n', ahmc_n_iterations);
fprintf('ahmc_n_burnin: %d\n', ahmc_n_burnin);
fprintf('samplezinit_n_iterations: %d\n', samplezinit_n_iterations);
fprintf('n_iterations: %d\n', n_iterations);
fprintf('---------------------------------------------------------------------------\n');

load(input_file_data, 'x', 't');
load(input_file_fold, 'idx_tr', 'idx_te');

% Preprocess the data.
% Normalize the training data to have zero mean and unit variance
if ~isempty(intersect(idx_tr, idx_te))
  error('Error in ''job_dpnnSampleZInterpolateAhmcWithStart_samplezinit_CV'': ''idx_tr'' and ''idx_te'' must not have same indices');  
end
if length(intersect(union(idx_tr, idx_te), 1:size(x, 1))) ~= size(x, 1)
  error('Error in ''job_dpnnSampleZInterpolateAhmcWithStart_samplezinit_CV'': ''idx_tr'' and ''idx_te'' do not contain all indices');   
end

x_tr = x(idx_tr, :);
x_te = x(idx_te, :);
t_tr = t(idx_tr, :);
t_te = t(idx_te, :);

x_mean = mean(x_tr, 1);
x_std = std(x_tr, 1);
x_std(x_std == 0) = 1;
x_tr = bsxfun(@times, bsxfun(@minus, x_tr, x_mean), 1 ./ x_std);
x_te = bsxfun(@times, bsxfun(@minus, x_te, x_mean), 1 ./ x_std);

% In case of regression also normalize the targets.
if strcmp(task, 'regress')
  t_mean = mean(t_tr, 1);
  t_std = std(t_tr, 1);
  t_std(t_std == 0) = 1;
  t_tr = bsxfun(@times, bsxfun(@minus, t_tr, t_mean), 1 ./ t_std);
  t_te = bsxfun(@times, bsxfun(@minus, t_te, t_mean), 1 ./ t_std);
end

ensemble = [];

for ii = 0:n_iterations
  output_file = sprintf('%s_%d.mat', output_file_prefix, ii);
  if exist(output_file, 'file') == 2
    % The file already exists and does not have to be computed
    fprintf('File ''%s'' already available\n', output_file);
    load(output_file, 'model', 'ahmc_out', 'rng_matlab', 'rng_mex');
    rng(rng_matlab);
    mexRngInit(rng_mex);
  else
    if ii == 0
      load(input_file_model, 'result');

      % Load initial model from sampleZ job
      model_init = result(samplezinit_n_iterations).model;
      rng_matlab = result(samplezinit_n_iterations).rng_matlab;
      rng_mex = result(samplezinit_n_iterations).rng_mex;
      rng(rng_matlab);
      mexRngInit(rng_mex);
      
      % Perform initial ahmc
      tic;
      func = @(w) funDpnnErrorFunction(w, model_init, x_tr, t_tr);
      grad = @(w) funDpnnErrorGradient(w, model_init, x_tr, t_tr);
      
      % Convert weights to vector format
      if strcmp(model_init.sharing, 'layerwise')
        w0 = zeros(1, sum(model_init.num_unique_weights));
        start_idx = 1;
        for ll = 1:model_init.num_layers
          end_idx = start_idx + model_init.num_unique_weights(ll) - 1;
          w0(start_idx:end_idx) = model_init.W{ll};
          start_idx = end_idx + 1;
        end
      elseif strcmp(model_init.sharing, 'global')
        w0 = model_init.W;
      else
        error('Error in ''job_dpnnSampleZInterpolateAhmcWithStart_samplezinit'': Unrecognized sharing ''%s''', model_init.sharing);  
      end
      ahmc_out = ahmcWithStart(ahmc_n_iterations, ahmc_n_burnin, [LF_min_eps, LF_max_eps; LF_min_L, LF_max_L], grad, func, sum(model_init.num_unique_weights), w0);
      time_needed_ahmc = toc;
      fprintf('Time needed [ahmc]: %f seconds\n', time_needed_ahmc);
      
      % Extract model from last ahmc weights
      model = model_init;
      if strcmp(model.sharing, 'layerwise')
        start_idx = 1;
        for ll = 1:model.num_layers
          end_idx = start_idx + model.num_unique_weights(ll) - 1;
          model.W{ll} = ahmc_out.saved(end, start_idx:end_idx);
          start_idx = end_idx + 1;
        end
      elseif strcmp(model.sharing, 'global')
        model.W = ahmc_out.saved(end, :);
      else
        error('Error in ''job_dpnnSampleZInterpolateAhmcWithStart_samplezinit_CV'': Unrecognized sharing ''%s''', model.sharing);  
      end
      
      % Extract state of RNG and save values
      rng_matlab = rng;
      rng_mex = mexRngState;
      save(output_file, 'model_init', 'model', 'ahmc_out', 'time_needed_ahmc', 'rng_matlab', 'rng_mex');
    else
      fprintf('Iteration %d/%d\n', ii, n_iterations);

      % Perform sampleZ
      model_init = model;
      tic;
      model_sampleZ = mexSampleZInterpolate(model_init, x_tr, t_tr, batch_size, discretization, 100, 2, interpolation_method);
      time_needed_sampleZ = toc;
      fprintf('Time needed [sampleZ]: %f seconds\n', time_needed_sampleZ);

      % Perform ahmc
      tic;
      func = @(w) funDpnnErrorFunction(w, model_sampleZ, x_tr, t_tr);
      grad = @(w) funDpnnErrorGradient(w, model_sampleZ, x_tr, t_tr);
      
      % Convert weights to vector format
      if strcmp(model_sampleZ.sharing, 'layerwise')
        w0 = zeros(1, sum(model_sampleZ.num_unique_weights));
        start_idx = 1;
        for ll = 1:model_sampleZ.num_layers
          end_idx = start_idx + model_sampleZ.num_unique_weights(ll) - 1;
          w0(start_idx:end_idx) = model_sampleZ.W{ll};
          start_idx = end_idx + 1;
        end
      elseif strcmp(model_sampleZ.sharing, 'global')
        w0 = model_sampleZ.W;
      else
        error('Error in ''job_dpnnSampleZInterpolateAhmcWithStart_samplezinit'': Unrecognized sharing ''%s''', model_sampleZ.sharing);  
      end
      ahmc_out = ahmcWithStart(ahmc_n_iterations, ahmc_n_burnin, [LF_min_eps, LF_max_eps; LF_min_L, LF_max_L], grad, func, sum(model_sampleZ.num_unique_weights), w0);
      time_needed_ahmc = toc;
      fprintf('Time needed [ahmc]: %f seconds\n', time_needed_ahmc);

      % Extract model from last ahmc weights
      model = model_sampleZ;
      if strcmp(model.sharing, 'layerwise')
        start_idx = 1;
        for ll = 1:model.num_layers
          end_idx = start_idx + model.num_unique_weights(ll) - 1;
          model.W{ll} = ahmc_out.saved(end, start_idx:end_idx);
          start_idx = end_idx + 1;
        end
      elseif strcmp(model.sharing, 'global')
        model.W = ahmc_out.saved(end, :);
      else
        error('Error in ''job_dpnnSampleZInterpolateAhmcWithStart_samplezinit_CV'': Unrecognized sharing ''%s''', model.sharing);  
      end

      % Extract state of RNG and save values
      rng_matlab = rng;
      rng_mex = mexRngState;
      save(output_file, 'model_init', 'model_sampleZ', 'model', 'ahmc_out', 'time_needed_sampleZ', 'time_needed_ahmc', 'rng_matlab', 'rng_mex');
    end
  end

  % Construct ensemble model for dpnnEvaluateEnsemble
  model_ensemble = model;
  if strcmp(model.sharing, 'layerwise')
    start_idx = 1;
    for ll = 1:model.num_layers
      end_idx = start_idx + model.num_unique_weights(ll) - 1;
      model_ensemble.W{ll} = ahmc_out.saved(:, start_idx:end_idx);
      start_idx = end_idx + 1;
    end
  elseif strcmp(model.sharing, 'global')
    model_ensemble.W = ahmc_out.saved;
  else
    error('Error in ''job_dpnnSampleZInterpolateAhmcWithStart_samplezinit_CV'': Unrecognized sharing ''%s''', model.sharing);  
  end
  ensemble = [ensemble, model_ensemble];
end

% Evaluate the ensemble.
output_file = sprintf('%s_%d_errors.mat', output_file_prefix, n_iterations);
if exist(output_file, 'file') == 2
  fprintf('File ''%s'' already available\n', output_file);
else
  if strcmp(task, 'regress')
    fprintf('Evaluating training data...\n');
    [errs_tr, lls_tr, lls_tr_all] = dpnnEvaluateRegressionEnsembleForward(ensemble, x_tr, t_tr, t_mean, t_std, true);
    fprintf('Evaluating test data...\n');
    [errs_te, lls_te, lls_te_all] = dpnnEvaluateRegressionEnsembleForward(ensemble, x_te, t_te, t_mean, t_std, true);
    save(output_file, 'errs_tr', 'errs_te', 'lls_tr', 'lls_te', 'lls_tr_all', 'lls_te_all');
  else
    fprintf('Evaluating training data...\n');
    errs_tr = dpnnEvaluateEnsembleForward(ensemble, x_tr, t_tr, true);
    fprintf('Evaluating test data...\n');
    errs_te = dpnnEvaluateEnsembleForward(ensemble, x_te, t_te, true);
    save(output_file, 'errs_tr', 'errs_te');
  end
end

fprintf('job_dpnnSampleZInterpolateAhmcWithStart_samplezinit_CV finished\n');

end

