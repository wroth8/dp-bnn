function grad = funNnErrorGradient(w, model, x, t)
% Wrapper that can be used for functions that expect a function handle of
% the gradient of the error function of plain feed-forward neural networks
% using vectorized weights.
%
% Input:
% w: The vectorized weights of the neural network.
% model: The neural network template. The weights of the model are replaced
%   by w.
% x: The input vectors stored in a matrix where rows correspond to
%   different samples and columns correspond to different features.
% t: The output vectors stored in a matrix where rows correspond to
%   different samples. In case the task of the model is 'regress' or
%   'biclass', the columns correspond to different outputs. In case the
%   task of the model is 'muclass', t must be a column vector containing
%   the class labels as integers starting from 1.
%
% Output:
% grad: The vectorized gradient of the error function with respect to the
%   weights (see nnErrorGradient)
%
% Example:
% grad = @(w) funNnErrorGradient(w, model, x, t);
%
% @author Wolfgang Roth

w_size = size(w);
w = w(:)';

start_idx = 1;
for l = 1:model.num_layers
  end_idx = start_idx + model.layout(l) * model.layout(l+1) - 1;
  model.W{l} = reshape(w(start_idx:end_idx), [model.layout(l), model.layout(l+1)]);

  start_idx = end_idx + 1;
  end_idx = start_idx + model.layout(l+1) - 1;
  model.b{l} = reshape(w(start_idx:end_idx), [1, model.layout(l+1)]);

  start_idx = end_idx + 1;
end

[grad_W, grad_b] = nnErrorGradient(model, x, t);

grad = zeros(size(w));
start_idx = 1;
for l = 1:model.num_layers
  end_idx = start_idx + model.layout(l) * model.layout(l+1) - 1;
  grad(start_idx:end_idx) = grad_W{l}(:);
  
  start_idx = end_idx + 1;
  end_idx = start_idx + model.layout(l+1) - 1;
  grad(start_idx:end_idx) = grad_b{l}(:);
  
  start_idx = end_idx + 1;
end

grad = reshape(grad, w_size);

end