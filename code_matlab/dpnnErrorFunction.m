function err = dpnnErrorFunction( model, x, t )
% Computes the error function of the DP neural network for the given data.
%
% Input:
% model: The DP neural network (see dpnnInitCRP)
% x: The input vectors stored in a matrix where rows correspond to
%   different samples and columns correspond to different features.
% t: The output vectors stored in a matrix where rows correspond to
%   different samples. In case the task of the model is 'regress' or
%   'biclass', the columns correspond to different outputs. In case the
%   task of the model is 'muclass', t must be a column vector containing
%   the class labels as integers starting from 1.
%
% Output:
% error: The error function of the neural network. The error depends on the
%  model's task:
%   'regress': Regularized squared error function
%   'biclass': Regularized cross entropy error function
%   'muclass': Regularized multiclass cross entropy error function
%
% Note: x and t can be given as gpuarray to utilize GPU computation
%
% @author Wolfgang Roth

% For compatibility with older code versions
if isnan(model.beta)
  model.beta = 1;
end

% Compute the outputs but not the activations. In case of classification
% the log of the sigmoid or the softmax activation needs to be computed
% with care in order to not get numerical problems.
for l = 1:model.num_layers
  if strcmp(model.sharing, 'layerwise')
    W = reshape(model.W{l}(model.ZW{l}), size(model.ZW{l}));
    b = reshape(model.W{l}(model.Zb{l}), size(model.Zb{l}));
  elseif strcmp(model.sharing, 'global')
    W = reshape(model.W(model.ZW{l}), size(model.ZW{l}));
    b = reshape(model.W(model.Zb{l}), size(model.Zb{l}));
  else
    error('Error in ''dpnnErrorFunction'': Unrecognized sharing ''%s''', model.sharing);
  end
  x = bsxfun(@plus, x * W, b);
  if l ~= model.num_layers
    if strcmp(model.activation, 'sigmoid')
      x = 1 ./ (1 + exp(-x));
    elseif strcmp(model.activation, 'tanh')
      x = tanh(x); %x = 1 - 2 ./ (1 + exp(2 * x));
    elseif strcmp(model.activation, 'relu')
      x = max(x, 0);
    else
      error('Error in ''dpnnErrorFunction'': Unrecognized activation ''%s''', model.task);      
    end
  end
end

% At this point x contains the outputs before applying the output
% activation function.
if strcmp(model.task, 'regress')
  err = sum(sum((x - t).^2)) / 2 / model.beta;
elseif strcmp(model.task, 'biclass')
  x = x * model.beta;
  % Compute log(1 + exp(-x)) with the log-sum-exp trick in order to avoid
  % numerical overflows.
  x1 = x(t == 1);
  x1 = x1(:);
  max_x = max(-x1, 0);
  err = sum(log(exp(-max_x) + exp(-x1-max_x)) + max_x);

  x0 = x(t == 0);
  x0 = x0(:);
  max_x = max(-x0, 0);
  err = err + sum(x0 + log(exp(-max_x) + exp(-x0-max_x)) + max_x);
elseif strcmp(model.task, 'muclass')
  x = x * model.beta;
  % Compute -log(exp(x_i) / sum exp(x_j)) with the log-sum-exp trick in
  % order to avoid numerical overflows.
  out_ind = sub2ind(size(x), 1:numel(t), t');
  err = -sum(x(out_ind));
  max_x = max(x, [], 2);
  x = bsxfun(@minus, x, max_x);
  x = exp(x);
  sum_x = sum(x, 2);
  sum_x = log(sum_x);
  err = err + sum(sum_x) + sum(max_x);
else
  error('Error in ''dpnnErrorFunction'': Unrecognized task ''%s''', model.task);
end

if strcmp(model.sharing, 'layerwise')
  for l = 1:model.num_layers
    err = err + model.W{l} * model.W{l}' / 2 / model.gamma(l);
  end
elseif strcmp(model.sharing, 'global')
  err = err + model.W * model.W' / 2 / model.gamma;
else
  error('Error in ''dpnnErrorFunction'': Unrecognized sharing ''%s''', model.sharing);
end

% In case GPU computing is used, transfer the error to the host
err = gather(err);

end
