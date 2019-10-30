function [errs, lls, lls_all, y] = nnEvaluateRegressionEnsemble( model, W, x, t, t_mean, t_std, varargin )
% This function is similar to nnEvaluateEnsemble but it is made especially
% for regression tasks where the target values have been normalized to
% zero-mean and unit variance. In addition it evaluates the log-likelihood
% by incrementally applying the log-sum-exp trick. The log-sum-exp trick is
% necessary since the probabilistic output of the ensemble can be seen as a
% large GMM where each neural network contributes with a mixture component.
% The errors and log-likelihoods are computed on the unnormalized targets.
%
% model: The model for which an ensemble should be averaged. This can be
%   seen as a template since its weights are essentially ignored and rather
%   replaced by weights stored in W.
% W: Matrix where each row corresponds to a full set of network weights.
% x: The input vectors stored in a matrix where rows correspond to
%   different samples and columns correspond to different features.
% t: The output vectors stored in a matrix where rows correspond to
%   different samples. The targets are already assumed to be normalized.
% t_mean: The mean value of the data that was used to normalize the
%   targets.
% t_std: The standard deviation of the data that was used to normalize the
%   targets.
% do_output [optional]: Determines whether some output about the progress
%   of the algorithm should be displayed or not (true or false, default:
%   true).
%
% Output:
% errs: Row vector containing as many elements as there are rows in W. Each
%   element contains the performance by averaging the outputs of the
%   corresonding network and all following networks. This has the advantage
%   that it is easier to throw away the predictions of leading models as
%   burn-in. The result by averaging all models can therefore be found in
%   errs(1).
% lls: Row vector containing the average log-likelihoods obtained by
%   evaluating the outputs of several models. The structure is similar as
%   that of errs.
% lls_all: Column vector of size N containing the log-likelihood of each
%   data sample obtained by averaging the whole ensemble.
% y: The averaged normalized output activations of the whole ensemble.
%
% Note: x and t can be given as gpuarray to utilize GPU computation
%
% @author Wolfgang Roth

% Check optional arguments
if length(varargin) > 1
  error('Error in ''nnEvaluateRegressionEnsemble'': Too much arguments given');
elseif length(varargin) == 1
  do_output = varargin{1};
else
  do_output = true;
end

if ~strcmp(model.task, 'regress')
  error('Error in ''nnEvaluateRegressionEnsemble'': The task must be ''regress''');
end

t_unnormalized = bsxfun(@plus, bsxfun(@times, t, t_std), t_mean);
beta_unnormalized = model.beta * t_std.^2;

n_models = size(W, 1);

y_cum = zeros(size(x, 1), model.layout(end));
e_cum = ones(size(x, 1), 1) * -inf;
errs = zeros(1, n_models);
lls = zeros(1, n_models);
D = size(t, 2);

for it = n_models:-1:1
  if do_output
    fprintf('Averaging last %d models (of %d models)\n', n_models - it + 1, n_models);
  end
  
  % Extracting weights and converting to neural network struct
  w = W(it, :);
  start_idx = 1;
  for l = 1:model.num_layers
    end_idx = start_idx + model.layout(l) * model.layout(l+1) - 1;
    model.W{l} = reshape(w(start_idx:end_idx), [model.layout(l), model.layout(l+1)]);
    
    start_idx = end_idx + 1;
    end_idx = start_idx + model.layout(l+1) - 1;
    model.b{l} = reshape(w(start_idx:end_idx), [1, model.layout(l+1)]);
    
    start_idx = end_idx + 1;
  end
  
  % Compute cumulative predictions and normalize (average)
  L = n_models - it + 1;
  y_sample = nnFunction(model, x);
  
  % Compute root mean squared error
  y_cum = y_cum + y_sample;
  y_unnormalized = bsxfun(@plus, bsxfun(@times, y_cum / L, t_std), t_mean);
  err = sqrt(sum(sum((t_unnormalized - y_unnormalized).^2)) / size(t, 1));
  
  % Compute log-likelihood
  y_sample = bsxfun(@plus, bsxfun(@times, y_sample, t_std), t_mean);
  e = -0.5 * sum(bsxfun(@times, (t_unnormalized - y_sample).^2, 1 ./ beta_unnormalized), 2);
  e_cum_max = max(e, e_cum);
  e_cum = log(exp(e_cum - e_cum_max) + exp(e - e_cum_max)) + e_cum_max; % log-sum-exp trick
  ll = e_cum - log(L) - 0.5 * D * log(2 * pi) - 0.5 * sum(log(beta_unnormalized));

  errs(it) = gather(err);
  lls(it) = gather(mean(ll));
end
lls_all = gather(ll);
y = y_cum / L;

end

