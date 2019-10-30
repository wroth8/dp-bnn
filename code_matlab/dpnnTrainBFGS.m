function [model, err, perf, fmincon_output] = dpnnTrainBFGS( model, x, t, n_iterations, varargin )
% Performs quasi-Newton optimization of the weights of a DP neural network
% with the Matlab builtin function fmincon. In case the number of weights
% to optimize a larger than 1024 the memory limited L-BFGS method is used.
% Otherwise the full BFGS algorithm is used.
%
% Input:
% model: The DP neural network (see dpnnInitCRP)
% x: The input vectors stored in a matrix where rows correspond to
%   different samples and columns correspond to different features. x can
%   also be given as cell array where each cell contains a separate
%   dataset (e.g. training, validation and test set). In this case t must
%   also be given as a cell array of the same size. The first element of
%   the cell array is used to optimize the model (i.e. the training set).
% t: The output vectors stored in a matrix where rows correspond to
%   different samples. In case the task of the model is 'regress' or
%   'biclass', the columns correspond to different outputs. In case the
%   task of the model is 'muclass', t must be a column vector containing
%   the class labels as integers starting from 1. In case x is given as a
%   cell array, t must also be given as a cell array where each cell
%   contains the targets of the corresponding dataset.
% n_iterations: The number of quasi-Newton updates to perform.
% do_output [optional]: Determines whether some output about the progress
%   of the algorithm should be displayed or not (true or false, default:
%   true).
%
% Output:
% model: The DP neural network with the optimized weights (see dpnnInitCRP)
% err: Contains the objective function values after each epoch (see
%   dpnnErrorFunction). In case more than a single dataset is given, i.e. x
%   and t are cell arrays, each row of err corresponds to a different
%   dataset.
% perf: Contains the performance of the model after each epoch (see
%  dpnnEvaluate). In case more than a single dataset is given, i.e. x and t
%   are cell arrays, each row of perf corresponds to a different dataset.
%
% Note: x and t can be given as gpuarray to utilize GPU computation
%
% @author Wolfgang Roth

% Check optional arguments
if length(varargin) > 1
  error('Error in ''dpnnTrainBFGS'': Too much arguments given');
elseif length(varargin) == 1
  do_output = varargin{1};
else
  do_output = true;
end

if (iscell(x) && ~iscell(t)) || (~iscell(x) && iscell(t))
  error('Error in ''dpnnTrainBFGS'': Either both ''x'' and ''t'' are cell arrays or none');
end

if ~iscell(x)
  x = {x};
  t = {t};
end

for it = 1:numel(x)
  % Targets are given - Check if input and target dimensions are consistent
  if size(x{it}, 1) ~= size(t{it}, 1)
    error('Error in ''dpnnTrainBFGS'': ''x{%d}'' and ''t{%d}'' must contain same number of rows', it, it);
  end
  if size(x{it}, 2) ~= model.layout(1)
    error('Error in ''dpnnTrainBFGS'': Number of columns in ''x{%d}'' are inconsistent with the given model', it);
  end
  if strcmp(model.task, 'muclass') && size(t{it}, 2) ~= 1
    error('Error in ''dpnnTrainBFGS'': Number of columns in ''t{%d}'' must be 1 in case of multiclass classification', it);
  elseif ~strcmp(model.task, 'muclass') && size(t{it}, 2) ~= model.layout(end)
    error('Error in ''dpnnTrainBFGS'': Number of columns in ''t{%d}'' are inconsistent with the given model', it);
  end
end

err = zeros(numel(x), n_iterations);
perf = zeros(numel(x), n_iterations);

% Convert weights to vector format
w0 = zeros(1, sum(model.num_unique_weights));
if strcmp(model.sharing, 'layerwise')
  start_idx = 1;
  for l = 1:model.num_layers
    end_idx = start_idx + model.num_unique_weights(l) - 1;
    w0(start_idx:end_idx) = model.W{l};
    start_idx = end_idx + 1;
  end
elseif strcmp(model.sharing, 'global')
  w0 = model.W;
else
  error('Error in ''dpnnTrainBFGS'': Unrecognized sharing ''%s''', model.sharing);
end

options = optimoptions('fmincon');
options = optimoptions(options, 'GradObj', 'on');
options = optimoptions(options, 'Algorithm', 'interior-point');

if length(w0) > 1024
  options = optimoptions(options, 'Hessian', {'lbfgs', 10});
else
  options = optimoptions(options, 'Hessian', 'bfgs');
end

options = optimoptions(options, 'MaxIter', n_iterations);
options = optimoptions(options, 'MaxFunEvals', n_iterations + ceil(n_iterations * 0.1 + 5000));
options = optimoptions(options, 'TolFun', 1e-10);
options = optimoptions(options, 'TolX', 1e-20);
options = optimoptions(options, 'OutputFcn', @outfun);

if do_output
  options = optimoptions(options, 'Display', 'iter');
else
  options = optimoptions(options, 'Display', 'off') ;
end

[~,~,~,fmincon_output] = fmincon(@objfun, w0, [], [], [], [], ones(size(w0)) * (-inf), [], [], options);

  function stop = outfun(vec_nn, optim_values, state)
    stop = 0;

    if optim_values.iteration == 0 || ~(strcmp(state, 'iter') || strcmp(state, 'done'))
      return
    end
      
    % Convert w and copy it to model
    if strcmp(model.sharing, 'layerwise')
      start_idx = 1;
      for l = 1:model.num_layers
        end_idx = start_idx + model.num_unique_weights(l) - 1;
        model.W{l} = vec_nn(start_idx:end_idx);
        start_idx = end_idx + 1;
      end
    else % strcmp(model.sharing, 'global')
      model.W = vec_nn;
    end
    
    err(1, optim_values.iteration) = optim_values.fval;
    perf(1, optim_values.iteration) = dpnnEvaluate(model, x{1}, t{1});
    for it_x = 2:numel(x)
      err(it_x, optim_values.iteration) = dpnnErrorFunction(model, x{it_x}, t{it_x});
      perf(it_x, optim_values.iteration) = dpnnEvaluate(model, x{it_x}, t{it_x});
    end
    
    if strcmp(state, 'done')
      err = err(:, 1:optim_values.iteration);
      perf = perf(:, 1:optim_values.iteration);
    end
  end

  function [f, g] = objfun(vec_nn)
    % Convert w and copy it to model
    if strcmp(model.sharing, 'layerwise')
      start_idx = 1;
      for l = 1:model.num_layers
        end_idx = start_idx + model.num_unique_weights(l) - 1;
        model.W{l} = vec_nn(start_idx:end_idx);
        start_idx = end_idx + 1;
      end
    else % strcmp(model.sharing, 'global')
      model.W = vec_nn;
    end

    % Compute function value and gradient
    [g, f] = dpnnErrorGradient(model, x{1}, t{1}, true);

    % Convert gradient to vector (there is nothing to do in case of global
    % sharing)
    if strcmp(model.sharing, 'layerwise')
      tmp_g = g;
      g = zeros(1, sum(model.num_unique_weights));
      start_idx = 1;
      for l = 1:model.num_layers
        end_idx = start_idx + model.num_unique_weights(l) - 1;
        g(start_idx:end_idx) = tmp_g{l};
        start_idx = end_idx + 1;
      end
    end
  end

end
