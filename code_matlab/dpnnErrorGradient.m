function [grad_W, err] = dpnnErrorGradient( model, x, t, varargin )
% Computes the gradient of the error function of the DP neural network for
% the given data using backpropagation.
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
% compute_error_func [optional]: Determines whether the error function
%   should be computed to save computation time if it is not needed (true
%   or false, default: true).
%
% Output:
% grad_W: The gradient with respect to the weights. In case of 'layerwise'
%   sharing grad_W is a cell array where each entry is a row vector
%   containing the gradient of the weights of the corresponding layer. In
%   case of 'global' sharing grad_W is a row vector containing the gradient
%   of the weights of the whole network.
% err: The error of the neural network (see dpnnErrorFunction). In case
%   compute_error_func is false, the output will be 0. Some procedures
%   require the gradient but not the error such as stochastic gradient
%   descent.
%
% Note: x and t can be given as gpuarray to utilize GPU computation
%
% @author Wolfgang Roth

% Check optional arguments
if length(varargin) > 1
  error('Error in ''dpnnErrorGradient'': Too much arguments given');
elseif length(varargin) == 1
  compute_error_func = varargin{1};
else
  compute_error_func = true;
end

% For compatibility with older code versions
if isnan(model.beta)
  model.beta = 1;
end

if strcmp(model.task, 'muclass')
  % Convert integer target class values to 0-1 matrix
  m = zeros(numel(t), model.layout(end));
  m(sub2ind(size(m), 1:numel(t), t')) = 1;
  t = m;
  clear m;
end

a = cell(1, model.num_layers);
z = cell(1, model.num_layers);
delta = cell(1, model.num_layers);

if strcmp(model.sharing, 'layerwise')
  grad_W = cell(1, model.num_layers);
  for l = 1:model.num_layers
    grad_W{l} = zeros(1, model.num_unique_weights(l));
  end
elseif strcmp(model.sharing, 'global')
  grad_W = zeros(1, model.num_unique_weights);
else
  error('Error in ''dpnnErrorGradient'': Unrecognized sharing ''%s''', model.sharing);
end

% Backpropagation: Forward pass
for l = 1:model.num_layers
  if strcmp(model.sharing, 'layerwise')
    W = reshape(model.W{l}(model.ZW{l}), size(model.ZW{l}));
    b = reshape(model.W{l}(model.Zb{l}), size(model.Zb{l}));
  elseif strcmp(model.sharing, 'global')
    W = reshape(model.W(model.ZW{l}), size(model.ZW{l}));
    b = reshape(model.W(model.Zb{l}), size(model.Zb{l}));
  else
    error('Error in ''dpnnErrorGradient'': Unrecognized sharing ''%s''', model.sharing);
  end

  if l == 1
    a{l} = bsxfun(@plus, x * W, b);
  else
    a{l} = bsxfun(@plus, z{l-1} * W, b);
  end
  
  if l ~= model.num_layers
    if strcmp(model.activation, 'sigmoid')
      z{l} = 1 ./ (1 + exp(-a{l}));
    elseif strcmp(model.activation, 'tanh')
      z{l} = tanh(a{l}); %z{l} = 1 - 2 ./ (1 + exp(2 * a{l}));
    elseif strcmp(model.activation, 'relu')
      z{l} = max(a{l}, 0);
    else
      error('Error in ''dpnnErrorGradient'': Unrecognized activation ''%s''', model.task);      
    end
  else
    if strcmp(model.task, 'regress')
      z{l} = a{l};
    elseif strcmp(model.task, 'biclass')
      z{l} = 1 ./ (1 + exp(-a{l} * model.beta));
    elseif strcmp(model.task, 'muclass')
      z{l} = a{l} * model.beta;
      z{l} = bsxfun(@minus, z{l}, max(z{l}, [], 2));
      z{l} = exp(z{l});
      z{l} = bsxfun(@times, z{l}, 1 ./ sum(z{l}, 2));
    else
      error('Error in ''dpnnErrorGradient'': Unrecognized task ''%s''', model.task);
    end
  end
end

% Backpropagation: Backward pass
for l = model.num_layers:-1:1
  if l == model.num_layers
    if strcmp(model.task, 'regress')
      delta{l} = (z{l} - t) / model.beta;
    else
      delta{l} = (z{l} - t) * model.beta;
    end
  else
    if strcmp(model.sharing, 'layerwise')
      W = reshape(model.W{l+1}(model.ZW{l+1}), size(model.ZW{l+1}));
    elseif strcmp(model.sharing, 'global')
      W = reshape(model.W(model.ZW{l+1}), size(model.ZW{l+1}));
    else
      error('Error in ''dpnnErrorGradient'': Unrecognized sharing ''%s''', model.sharing);
    end

    if strcmp(model.activation, 'sigmoid')
      delta{l} = (delta{l+1} * W') .* (z{l} .* (1-z{l}));
    elseif strcmp(model.activation, 'tanh')
      delta{l} = (delta{l+1} * W') .* (1 - z{l}.^2);
    elseif strcmp(model.activation, 'relu')
      delta{l} = (delta{l+1} * W') .* (a{l} > 0);  
    else
      error('Error in ''dpnnErrorGradient'': Unrecognized activation ''%s''', model.task); 
    end
  end
  
  % Compute gradient for normal weights
  if l == 1
    grad_W_tmp = x' * delta{l};
  else
    grad_W_tmp = z{l-1}' * delta{l};
  end
  grad_b_tmp = sum(delta{l}, 1);

  % Accumulate gradients of shared connections
  if strcmp(model.sharing, 'global')
    grad_W_tmp = gather(grad_W_tmp);
    grad_b_tmp = gather(grad_b_tmp);
    grad_W = grad_W + accumarray(model.ZW{l}(:), grad_W_tmp(:), [model.num_unique_weights, 1])' + accumarray(model.Zb{l}(:), grad_b_tmp(:), [model.num_unique_weights, 1])';
  elseif strcmp(model.sharing, 'layerwise')
    grad_W_tmp = gather(grad_W_tmp);
    grad_b_tmp = gather(grad_b_tmp);
    grad_W{l} = accumarray(model.ZW{l}(:), grad_W_tmp(:), [model.num_unique_weights(l), 1])' + accumarray(model.Zb{l}(:), grad_b_tmp(:), [model.num_unique_weights(l), 1])';
    grad_W{l} = grad_W{l} + model.W{l} / model.gamma(l);
  else
    error('Error in ''dpnnErrorGradient'': Unrecognized sharing ''%s''', model.sharing);
  end
end

if strcmp(model.sharing, 'global')
  grad_W = grad_W + model.W / model.gamma;
end

% a{model.num_layers} contains the activations of the last layer before
% applying the output activation function. We can use these to compute the
% error function if needed.
if compute_error_func
  x = a{model.num_layers};
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
    err = -sum(x(t == 1));
    max_x = max(x, [], 2);
    x = bsxfun(@minus, x, max_x);
    x = exp(x);
    sum_x = sum(x, 2);
    sum_x = log(sum_x);
    err = err + sum(sum_x) + sum(max_x);
  else
    error('Error in ''dpnnErrorGradient'': Unrecognized task ''%s''', model.task);
  end

  if strcmp(model.sharing, 'layerwise')
    for l = 1:model.num_layers
      err = err + model.W{l} * model.W{l}' / 2 / model.gamma(l);
    end
  elseif strcmp(model.sharing, 'global')
    err = err + model.W * model.W' / 2 / model.gamma;
  else
    error('Error in ''dpnnErrorGradient'': Unrecognized sharing ''%s''', model.sharing);
  end
  err = gather(err);
else
  err = 0;
end

end

