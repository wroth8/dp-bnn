function [ grad_W, grad_b, err ] = nnErrorGradient( model, x, t, varargin )
% Computes the gradient of the error function of the neural network for the
% given data using backpropagation.
%
% Input:
% model: The neural network (see nnInit)
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
% grad_W: Cell array containing the gradient of the weights of each layer.
% grad_b: Cell array containing the gradient of the biases of each layer.
% err: The error of the neural network (see nnErrorFunction). In case
%   compute_error_func is false, the output will be 0. Some procedures
%   require the gradient but not the error such as stochastic gradient
%   descent.
%
% Note: x and t can be given as gpuarray to utilize GPU computation
%
% @author Wolfgang Roth

% Check optional arguments
if length(varargin) > 1
  error('Error in ''nnErrorGradient'': Too much arguments given');
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

grad_W = cell(1, model.num_layers);
grad_b = cell(1, model.num_layers);

% Backpropagation: Forward pass
for l = 1:model.num_layers
  if l == 1
    a{l} = bsxfun(@plus, x * model.W{l}, model.b{l});
  else
    a{l} = bsxfun(@plus, z{l-1} * model.W{l}, model.b{l});
  end
  
  if l ~= model.num_layers
    if strcmp(model.activation, 'sigmoid')
      z{l} = 1 ./ (1 + exp(-a{l}));
    elseif strcmp(model.activation, 'tanh')
      z{l} = tanh(a{l}); %z{l} = 1 - 2 ./ (1 + exp(2 * a{l}));
    elseif strcmp(model.activation, 'relu')
      z{l} = max(a{l}, 0);
    else
      error('Error in ''nnErrorGradient'': Unrecognized activation ''%s''', model.task);      
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
      error('Error in ''nnErrorGradient'': Unrecognized task ''%s''', model.task);
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
    if strcmp(model.activation, 'sigmoid')
      delta{l} = (delta{l+1} * model.W{l+1}') .* (z{l} .* (1-z{l}));
    elseif strcmp(model.activation, 'tanh')
      delta{l} = (delta{l+1} * model.W{l+1}') .* (1 - z{l}.^2);
    elseif strcmp(model.activation, 'relu')
      delta{l} = (delta{l+1} * model.W{l+1}') .* (a{l} > 0);  
    else
      error('Error in ''nnErrorGradient'': Unrecognized activation ''%s''', model.task); 
    end
  end
  
  % Compute gradient for normal weights
  if l == 1
    grad_W{l} = x' * delta{l};
  else
    grad_W{l} = z{l-1}' * delta{l};
  end
  grad_b{l} = sum(delta{l}, 1);
  
  grad_W{l} = grad_W{l} + model.W{l} / model.gamma;
  grad_b{l} = grad_b{l} + model.b{l} / model.gamma;
end

grad_W = cellfun(@(x) gather(x), grad_W, 'UniformOutput', false);
grad_b = cellfun(@(x) gather(x), grad_b, 'UniformOutput', false);

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
    error('Error in ''nnErrorGradient'': Unrecognized task ''%s''', model.task);
  end

  err = err + (sum(cellfun(@(w) sum(w(:).^2), model.W)) + sum(cellfun(@(b) sum(b.^2), model.b))) * 0.5 / model.gamma;
  err = gather(err);
else
  err = 0;
end

end

