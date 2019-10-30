function [errs, y] = dpnnEvaluateEnsemble( models, x, t, varargin )
% Evaluates an ensemble of DP neural networks and averages their outputs to
% compute a prediction.
%
% Input:
% models: Row vector of DP neural networks (see dpnnInitCRP). For each
%   model the weight vectors models(i).W can be given as matrices such that
%   each row corresponds to a different network with the same sharing
%   configuration but a different set of weights. It is assumed that the
%   models are sampled in the following linear order: First, the elements
%   in models are sampled starting at models(1). Second, within each
%   element of models the samples are drawn starting from the first row of
%   models(i).W. This is important for how the results are averaged.
% x: The input vectors stored in a matrix where rows correspond to
%   different samples and columns correspond to different features.
% t: The output vectors stored in a matrix where rows correspond to
%   different samples. In case the task of the model is 'regress' or
%   'biclass', the columns correspond to different outputs. In case the
%   task of the model is 'muclass', t must be a column vector containing
%   the class labels as integers starting from 1.
% do_output [optional]: Determines whether some output about the progress
%   of the algorithm should be displayed or not (true or false, default:
%   true).
%
% Output:
% errs: Cell array with length(models) entries containing the predictions
%   by averaging the outputs of several models. Cell i contains a row
%   vector of the same length as there are rows in the corresponding
%   models(i).W. Each element in a cell contains the performance by
%   averaging the outputs of the corresonding sample and all *following*
%   samples. This has the advantage that it is easier to investigate the
%   influence of burn-in as the error for a certain burn-in can be obtained
%   by just omitting the first n_burnin samples, i.e., accessing the
%   n_burnin-th element of cat(2, errs{:}). The result by averaging all
%   models can therefore be found in errs{1}(1).
% y: The averaged output activations of the whole ensemble.
%
% Note: x and t can be given as gpuarray to utilize GPU computation
%
% @author Wolfgang Roth

% Check optional arguments
if length(varargin) > 1
  error('Error in ''dpnnEvaluateEnsemble'': Too much arguments given');
elseif length(varargin) == 1
  do_output = varargin{1};
else
  do_output = true;
end

if ~isrow(models) && ~isstruct(models)
  error('Error in ''dpnnEvaluateEnsemble'': Argument ''models'' must be a row vector of structs');
end

y_cum = zeros(size(x, 1), models(1).layout(end));
y_cum_count = 0;
errs = cell(1, length(models));
for it_models = length(models):-1:1
  if do_output
    fprintf('Processing configuration %d/%d\n', length(models)-it_models+1, length(models));
  end
  model = models(it_models);

  % Extract samples and check consistency
  W = model.W;
  if strcmp(model.sharing, 'layerwise')
    n_samples = size(W{1}, 1);
    for l = 1:model.num_layers
      model.W{l} = [];
      if size(W{l}, 1) ~= n_samples
        error('Error in ''dpnnEvaluateEnsemble'': Inconsistent weight matrices');
      end
    end
  elseif strcmp(model.sharing, 'global')
    n_samples = size(W, 1);
    model.W = [];
  else
    error('Error in ''dpnnEvaluateEnsemble'': Unrecognized sharing ''%s''', model.sharing);
  end
  errs{it_models} = zeros(1, n_samples);

  for it_sample = n_samples:-1:1
    if do_output
      fprintf('Processing sample %d/%d\n', n_samples-it_sample+1, n_samples);
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
    y_cum = y_cum + dpnnFunction(model, x);
    y_cum_count = y_cum_count + 1;
    y = y_cum / y_cum_count;

    if strcmp(model.task, 'regress')
      % Root mean squared error
      err = sqrt(sum(sum((t - y).^2)) / size(t, 1));
    elseif strcmp(model.task, 'biclass')
      % Mean classification rate per output
      y = round(y);
      err = sum(sum(abs(t - y))) / numel(t);
    elseif strcmp(model.task, 'muclass')
      % Mean classification rate
      [~, y_idx] = max(y, [], 2);
      err = sum(t ~= y_idx) / size(t, 1);
    else
      error('Error in ''dpnnEvaluateEnsemble'': Unrecognized task ''%s''', model.task);
    end
    errs{it_models}(it_sample) = gather(err);
  end
end

y = y_cum / y_cum_count;

end

