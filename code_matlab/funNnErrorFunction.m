function err = funNnErrorFunction(w, model, x, t)
% Wrapper that can be used for functions that expect a function handle of
% the error function of plain feed-forward neural networks using vectorized
% weights.
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
% err: The value of the error function (see nnErrorFunction)
%
% Example:
% func = @(w) funNnErrorFunction(w, model, x, t);
%
% @author Wolfgang Roth

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

err = nnErrorFunction(model, x, t);

end