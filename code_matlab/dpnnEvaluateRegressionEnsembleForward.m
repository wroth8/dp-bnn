function [errs, lls, lls_all, y] = dpnnEvaluateRegressionEnsembleForward( models, x, t, t_mean, t_std, varargin )
% This function is basically the same as dpnnEvaluateRegressionEnsemble
% with the only difference that the cumulative sample averages are computed
% in the order as they are given in models. This is more natural than the
% reverse order, but it does not allow easily to verify the influence of
% burn-in which is more easily done via dpnnEvaluateRegressionEnsemble.
% Some more details can be found in the documentation of
% dpnnEvaluateRegressionEnsembleForward.
%
% Input:
% models: Row vector of DP neural networks (see dpnnInitCRP). For each
%   model the weight vectors models(i).W can be given as matrices such that
%   each row corresponds to a different network with the same sharing
%   configuration but a different set of weights. It is assumed that the
%   models are sampled in the following linear order: First, the elements
%   in models are sampled starting at models(1). Second, within each
%   element of models the samples are drawn starting from the first row of
%   models(i).W. This is important for how the results are averaged. The
%   task must be set to 'regress', otherwise an error will show up.
% x: The input vectors stored in a matrix where rows correspond to
%   different samples and columns correspond to different features.
% t: The output vectors stored in a matrix where rows correspond to
%   different samples. The targets are already assumed to be normalized.
% t_mean: The mean value of the data that was used to normalize the
%   targets.
% t_std: The standard deviation of the data that was used to normalize the
%   targets.
% do_output [optional]: Determines whether some output about the progress
%   of the algorithm should be displayed or not (true or false, default:
%   true).
%
% Output:
% errs: Cell array with length(models) entries containing the predictions
%   by averaging the outputs of several models. Cell i contains a row
%   vector of the same length as there are rows in the corresponding
%   models(i).W. Each element in a cell contains the performance by
%   averaging the outputs of the corresonding sample and all preceding
%   samples. The result by averaging all models can therefore be found in
%   errs{end}(end). Note that the order in which samples are averaged is
%   essentially the difference to dpnnEvaluateRegressionEnsemble.
% lls: Cell array with length(models) entries containing the average
%   log-likelihoods obtained by evaluating the outputs of several models.
%   The structure is similar as that of errs.
% lls_all: Matrix of size N x length(models) containing the average
%   log-likelihoods of the individual data samples. Column i contains the
%   log-likelihoods obtained by averaging all models of the corresponding
%   struct entry and all following models. The log-likelihoods obtained by
%   averaging all models can therefore be found in lls_all(:,end).
% y: The averaged normalized output activations of the whole ensemble.
%
% Note: x and t can be given as gpuarray to utilize GPU computation
%
% @author Wolfgang Roth

% Check optional arguments
if length(varargin) > 1
  error('Error in ''dpnnEvaluateRegressionEnsembleForward'': Too much arguments given');
elseif length(varargin) == 1
  do_output = varargin{1};
else
  do_output = true;
end

if ~isrow(models) && ~isstruct(models)
  error('Error in ''dpnnEvaluateRegressionEnsembleForward'': Argument ''models'' must be a row vector of structs');
end

if ~strcmp(models(1).task, 'regress')
  error('Error in ''dpnnEvaluateRegressionEnsembleForward'': The task must be ''regress''');
end

t_unnormalized = bsxfun(@plus, bsxfun(@times, t, t_std), t_mean);

y_cum = zeros(size(x, 1), models(1).layout(end));
errs = cell(1, length(models));

lls = cell(1, length(models));
lls_all = zeros(size(x,1), length(models));
e_cum = ones(size(x,1), 1) * -inf;
L = 0;
D = size(t, 2);

for it_models = 1:length(models)
  if do_output
    fprintf('Processing configuration %d/%d\n', it_models, length(models));
  end
  model = models(it_models);

  % Extract samples and check consistency
  W = model.W;
  if strcmp(model.sharing, 'layerwise')
    n_samples = size(W{1}, 1);
    for l = 1:model.num_layers
      model.W{l} = [];
      if size(W{l}, 1) ~= n_samples
        error('Error in ''dpnnEvaluateRegressionEnsembleForward'': Inconsistent weight matrices');
      end
    end
  elseif strcmp(model.sharing, 'global')
    n_samples = size(W, 1);
    model.W = [];
  else
    error('Error in ''dpnnEvaluateRegressionEnsembleForward'': Unrecognized sharing ''%s''', model.sharing);
  end
  errs{it_models} = zeros(1, n_samples);
  lls{it_models} = zeros(1, n_samples);
  beta_unnormalized = model.beta * t_std.^2;

  for it_sample = 1:n_samples
    if do_output
      fprintf('Processing sample %d/%d\n', it_sample, n_samples);
    end

    % Extract sample
    if strcmp(model.sharing, 'layerwise')
      for l = 1:model.num_layers
        model.W{l} = W{l}(it_sample, :);
      end
    else % strcmp(model.sharing, 'global')
      model.W = W(it_sample, :);
    end

    % Compute cumulative predictions and normalize (average)
    y_sample = dpnnFunction(model, x);
    L = L + 1;
    
    % Compute root mean squared error
    y_cum = y_cum + y_sample;
    y_unnormalized = bsxfun(@plus, bsxfun(@times, y_cum / L, t_std), t_mean);
    err = sqrt(sum(sum((t_unnormalized - y_unnormalized).^2)) / size(t, 1));
    errs{it_models}(it_sample) = gather(err);
    
    % Compute log-likelihood
    y_sample = bsxfun(@plus, bsxfun(@times, y_sample, t_std), t_mean);
    e = -0.5 * sum(bsxfun(@times, (t_unnormalized - y_sample).^2, 1 ./ beta_unnormalized), 2) - 0.5 * sum(log(beta_unnormalized));
    e_cum_max = max(e, e_cum);
    e_cum = log(exp(e_cum - e_cum_max) + exp(e - e_cum_max)) + e_cum_max; % log-sum-exp trick
    
    ll = e_cum - log(L) - 0.5 * D * log(2 * pi);
    ll_mean = mean(ll);
    lls{it_models}(it_sample) = gather(ll_mean);
  end
  lls_all(:, it_models) = gather(ll);
end

y = y_cum / L;

end

