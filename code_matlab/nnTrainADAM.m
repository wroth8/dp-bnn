function [model, err, perf] = nnTrainADAM( model, x, t, batch_size, step_size, n_epochs, varargin )
% Performs stochastic gradient descent with the ADAM algorithm [1] to
% optimize the weights of a plain neural network. Optimization is stopped
% prematurely if the objective function changes less than 1e-6 within
% 5 epochs.
%
% Input:
% model: The neural network (see nnInit)
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
% batch_size: The number of samples in a batch which is used to compute an
%   approximate gradient. In each epoch all samples are processed exactly
%   once but the batches are chosen randomly.
% step_size: The initial step step_size determining the influence of the
%   gradient in each step.
% n_epochs: The number of epochs of the optimization. An epoch consists of
%   possibly several weight updates where all together each sample is
%   processed exactly once.
% do_output [optional]: Determines whether some output about the progress
%   of the algorithm should be displayed or not (true or false, default:
%   true).
%
% Output:
% model: The neural network with the optimized weights (see nnInit)
% err: Contains the objective function values after each epoch (see
%   nnErrorFunction). In case more than a single dataset is given, i.e. x
%   and t are cell arrays, each row of err corresponds to a different
%   dataset.
% perf: Contains the performance of the model after each epoch (see
%  nnEvaluate). In case more than a single dataset is given, i.e. x
%   and t are cell arrays, each row of perf corresponds to a different
%   dataset.
%
% Note: x and t can be given as gpuarray to utilize GPU computation
%
% @author Wolfgang Roth

% Check optional arguments
if length(varargin) > 1
  error('Error in ''nnTrainADAM'': Too much arguments given');
elseif length(varargin) == 1
  do_output = varargin{1};
else
  do_output = true;
end

if (iscell(x) && ~iscell(t)) || (~iscell(x) && iscell(t))
  error('Error in ''nnTrainADAM'': Either both ''x'' and ''t'' are cell arrays or none');
end

if ~iscell(x)
  x = {x};
  t = {t};
end

for it = 1:numel(x)
  % Targets are given - Check if input and target dimensions are consistent
  if size(x{it}, 1) ~= size(t{it}, 1)
    error('Error in ''nnTrainADAM'': ''x{%d}'' and ''t{%d}'' must contain same number of rows', it, it);
  end
  if size(x{it}, 2) ~= model.layout(1)
    error('Error in ''nnTrainADAM'': Number of columns in ''x{%d}'' are inconsistent with the given model', it);
  end
  if strcmp(model.task, 'muclass') && size(t{it}, 2) ~= 1
    error('Error in ''nnTrainADAM'': Number of columns in ''t{%d}'' must be 1 in case of multiclass classification', it);
  elseif ~strcmp(model.task, 'muclass') && size(t{it}, 2) ~= model.layout(end)
    error('Error in ''nnTrainADAM'': Number of columns in ''t{%d}'' are inconsistent with the given model', it);
  end
end

err = zeros(numel(x), n_epochs);
perf = zeros(numel(x), n_epochs);

N_tr = size(x{1}, 1);
f = nnErrorFunction(model, x{1}, t{1});

alpha = step_size;
beta1 = 0.9;
beta2 = 0.999;
beta1t = 1;
beta2t = 1;
eps = 1e-8;
m_W = cellfun(@(x) zeros(size(x)), model.W, 'UniformOutput', false);
v_W = cellfun(@(x) zeros(size(x)), model.W, 'UniformOutput', false);
m_b = cellfun(@(x) zeros(size(x)), model.b, 'UniformOutput', false);
v_b = cellfun(@(x) zeros(size(x)), model.b, 'UniformOutput', false);

if do_output
  fprintf('nnTrainADAM\n');
  fprintf('init: f=%10.4f   step_size=%e\n', f, step_size);
end

for it = 1:n_epochs
  % Shuffle the training set
  shuffle_permutation = randperm(N_tr);
  x{1} = x{1}(shuffle_permutation, :);
  t{1} = t{1}(shuffle_permutation, :);
  
  % Perform batch gradient descent
  for it_n = 1:batch_size:N_tr
    x_batch = x{1}(it_n:min(it_n + batch_size - 1, N_tr), :);
    t_batch = t{1}(it_n:min(it_n + batch_size - 1, N_tr), :);
    N_batch = size(x_batch, 1);
    [grad_batch_W, grad_batch_b] = nnErrorGradient(model, x_batch, t_batch, false);
    for l = 1:model.num_layers
      % Gradient for likelihood term must be weighted according to batch size
      grad_batch_W{l} = grad_batch_W{l} - model.W{l} / model.gamma;
      grad_batch_W{l} = grad_batch_W{l} * N_tr / N_batch;
      grad_batch_W{l} = grad_batch_W{l} + model.W{l} / model.gamma;
      
      grad_batch_b{l} = grad_batch_b{l} - model.b{l} / model.gamma;
      grad_batch_b{l} = grad_batch_b{l} * N_tr / N_batch;
      grad_batch_b{l} = grad_batch_b{l} + model.b{l} / model.gamma;

      m_W{l} = beta1 * m_W{l} + (1 - beta1) * grad_batch_W{l};
      v_W{l} = beta2 * v_W{l} + (1 - beta2) * grad_batch_W{l}.^2;
      
      m_b{l} = beta1 * m_b{l} + (1 - beta1) * grad_batch_b{l};
      v_b{l} = beta2 * v_b{l} + (1 - beta2) * grad_batch_b{l}.^2;
      
      beta1t = beta1t * beta1;
      beta2t = beta2t * beta2;
      m_W_hat = m_W{l} ./ (1 - beta1t);
      v_W_hat = v_W{l} ./ (1 - beta2t);
      m_b_hat = m_b{l} ./ (1 - beta1t);
      v_b_hat = v_b{l} ./ (1 - beta2t);
      
      model.W{l} = model.W{l} - alpha * m_W_hat ./ (sqrt(v_W_hat) + eps);
      model.b{l} = model.b{l} - alpha * m_b_hat ./ (sqrt(v_b_hat) + eps);
    end
  end
  
  % Evaluate network on given data
  for it_x = 1:numel(x)
    err(it_x, it) = nnErrorFunction(model, x{it_x}, t{it_x});
    perf(it_x, it) = nnEvaluate(model, x{it_x}, t{it_x});
  end

  if do_output
    fprintf('%4d: f=%10.4f\n', it, err(1, it));
  end
  
  % Check if some stopping criterion is met
  if it >= 5
    % Check if the difference between the maximum and minimum function
    % value within the last five epochs is less than 1e-6
    min_f = min(err(1, (it-4):it));
    max_f = max(err(1, (it-4):it));
    if max_f - min_f < 1e-6
      fprintf('Stopping SGD (ADAM) optimization due to negligible changes in function value within the last five epochs\n');
      err = err(:, 1:it);
      perf = perf(:, 1:it);
      break;
    end
  end
end

end
