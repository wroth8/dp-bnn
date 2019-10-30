function x = dpnnPrediction( model, x )
% Computes the prediction of the given neural network model for the given
% inputs.
%
% Input:
% model: The DP neural network (see dpnnInitCRP)
% x: The inputs to the network. Each row corresponds to a data sample. Each
%   column corresponds to a feature.
%
% Output:
% x: The predictions of the network. The type of output depends on the
%   given model's task:
%   'regress': Same output as dpnnFunction
%   'biclass': Rounds the outputs of dpnnFunction
%   'muclass': The index of the largest value in each row of the
%     dpnnFunction output
%
% Note: x can be given as gpuarray to utilize GPU computation
%
% @author Wolfgang Roth

x = dpnnFunction(model, x);
if strcmp(model.task, 'biclass')
  x = round(x);
elseif strcmp(model.task, 'muclass')
  [~, x] = max(x, [], 2);
elseif ~strcmp(model.task, 'regress')
  error('Error in ''dpnnPrediction'': Unrecognized task ''%s''', model.task);
end


end

