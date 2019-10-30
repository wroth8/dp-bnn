function x = dpnnFunction( model, x )
% Computes the function of the given DP neural network for the given
% inputs.
%
% Input:
% model: The DP neural network (see dpnnInitCRP)
% x: The input vectors stored in a matrix where rows correspond to
%   different samples and columns correspond to different features.
%
% Output:
% x: The output of the network. Each row is the output for the
%   corresponding row of the input vector x. Each column corresponds to an
%   output dimension. In case the task of the model is 'muclass', the
%   output contains a vector with non-negative entries which sum to one.
%
% Note: x can be given as gpuarray to utilize GPU computation
%
% @author Wolfgang Roth

% For compatibility with older code versions
if isnan(model.beta)
  model.beta = 1;
end

for l = 1:model.num_layers
  if strcmp(model.sharing, 'layerwise')
    W = reshape(model.W{l}(model.ZW{l}), size(model.ZW{l}));
    b = reshape(model.W{l}(model.Zb{l}), size(model.Zb{l}));
  elseif strcmp(model.sharing, 'global')
    W = reshape(model.W(model.ZW{l}), size(model.ZW{l}));
    b = reshape(model.W(model.Zb{l}), size(model.Zb{l}));
  else
    error('Error in ''dpnnFunction'': Unrecognized sharing ''%s''', model.sharing);
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
      error('Error in ''dpnnFunction'': Unrecognized activation ''%s''', model.task);      
    end
  else
    if strcmp(model.task, 'biclass')
      x = 1 ./ (1 + exp(-x * model.beta));
    elseif strcmp(model.task, 'muclass')
      x = x * model.beta;
      x = bsxfun(@minus, x, max(x, [], 2));
      x = exp(x);
      x = bsxfun(@times, x, 1 ./ sum(x, 2));
    elseif ~strcmp(model.task, 'regress')
      error('Error in ''dpnnFunction'': Unrecognized task ''%s''', model.task);
    end
  end
end

x = gather(x);

end

