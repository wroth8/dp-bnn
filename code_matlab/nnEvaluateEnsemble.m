function errs = nnEvaluateEnsemble( model, W, x, t, varargin )
% Evaluates an ensemble of plain feed-forward neural networks with the same
% layout and averages their outputs to compute a prediction.
%
% model: The model for which an ensemble should be averaged. This can be
%   seen as a template since its weights are essentially ignored and rather
%   replaced by weights stored in W.
% W: Matrix where each row corresponds to a full set of network weights.
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
% errs: Row vector containing as many elements as there are rows in W. Each
%   element contains the performance by averaging the outputs of the
%   corresonding network and all following networks. This has the advantage
%   that it is easier to throw away the predictions of leading models as
%   burn-in. The result by averaging all models can therefore be found in
%   errs(1).
%
% Note: x and t can be given as gpuarray to utilize GPU computation
%
% @author Wolfgang Roth

% Check optional arguments
if length(varargin) > 1
  error('Error in ''nnEvaluateEnsemble'': Too much arguments given');
elseif length(varargin) == 1
  do_output = varargin{1};
else
  do_output = true;
end

n_models = size(W, 1);

y_cum = zeros(size(x, 1), model.layout(end));
errs = zeros(1, n_models);

for it = n_models:-1:1
  if do_output
    fprintf('Averaging last %d models (of %d models)\n', n_models - it + 1, n_models);
  end
  
  % Extracting weights and converting to neural network struct
  w = W(it, :);
  start_idx = 1;
  for l = 1:model.num_layers
    end_idx = start_idx + model.layout(l) * model.layout(l+1) - 1;
    model.W{l} = reshape(w(start_idx:end_idx), [model.layout(l), model.layout(l+1)]);
    
    start_idx = end_idx + 1;
    end_idx = start_idx + model.layout(l+1) - 1;
    model.b{l} = reshape(w(start_idx:end_idx), [1, model.layout(l+1)]);
    
    start_idx = end_idx + 1;
  end
  
  % Compute cumulative predictions and normalize (average)
  y_cum = y_cum + nnFunction(model, x);
  y = y_cum / (n_models - it + 1);
  
  if strcmp(model.task, 'regress')
    % Mean squared error
    err = sum(sum((t - y).^2)) / size(t, 1);
  elseif strcmp(model.task, 'biclass')
    % Mean classification rate per output
    y = round(y);
    err = sum(sum(abs(t - y))) / numel(t);
  elseif strcmp(model.task, 'muclass')
    % Mean classification rate
    [~, y_idx] = max(y, [], 2);
    err = sum(t ~= y_idx) / size(t, 1);
  else
    error('Error in ''nnEvaluateEnsemble'': Unrecognized task ''%s''', model.task);
  end
  errs(it) = gather(err);
end

end

