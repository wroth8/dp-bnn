function err = nnEvaluate( model, x, t )
% Evaluates the given neural network and returns the performance on the
% given data.
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
%
% Output:
% err: The error of the given model for input x and targets t. The error
%   depends on the model's task:
%    'regress': The mean squared error
%    'biclass': The mean classification error per output
%    'muclass': The classification error
%
% Note: x and t can be given as gpuarray to utilize GPU computation
%
% @author Wolfgang Roth

y = nnFunction(model, x);
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
  error('Error in ''nnEvaluate'': Unrecognized task ''%s''', model.task);
end

err = gather(err);

end

