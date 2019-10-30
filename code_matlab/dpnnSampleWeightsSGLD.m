function [ out, last_model ] = dpnnSampleWeightsSGLD( model, x, t, batch_size, num_iterations, num_samples_burnin, step_size_a, step_size_b, step_size_gamma, update_step_size_epoch, varargin )
% TODO: Documentation
% A sample is collected after every epoch
% schedule for step size: step_size_a / (step_size_b + t_epoch)^gamma , \gamma \in (0.5, 1]
% Recommendations for a,b,gamma: b=1,gamma=0.55
%
% Output:
% out: Struct containing information about the hybrid Monte Carlo algorithm
%   'saved': Matrix containing the intermediate sampled values of the
%     hybrid Monte Carlo algorithm. Each row corresponds to a drawn sample.
%     Note, that the initial values are not stored.
%   'squared_distance': Row vector containing the squared distance of
%     consecutive samples.
%   'accept_prob': Row vector containing the acceptance probability of the
%     corresponding sample.
% last_model: The last sampled model as DP neural network struct (see
%   dpnnInitCRP).
%
% @author Wolfgang Roth
% @reference M. Welling and Y.W.Teh., Bayesian Learning via Stochastic
%   Gradient Langevin Dynamics, ICML 2011


% Check optional arguments
if length(varargin) > 1
  error('Error in ''dpnnSampleSGLD'': Too much arguments given');
elseif length(varargin) == 1
  do_output = varargin{1};
else
  do_output = true;
end

N_tr = size(x, 1);

num_samples = num_iterations - num_samples_burnin;
out.saved = zeros(num_samples, sum(model.num_unique_weights));
out.squared_distance = zeros(1, num_samples);

n_samples_drawn = 0; % Number of samples
for it = 1:num_iterations
  % Shuffle the training set
  shuffle_permutation = randperm(N_tr);
  x = x(shuffle_permutation, :);
  t = t(shuffle_permutation, :);
  
  if update_step_size_epoch
    % Update the step_size after every epoch
    step_size = step_size_a / (step_size_b + it - 1)^(step_size_gamma);
  end
  
  % Perform batch gradient descent
  for it_n = 1:batch_size:N_tr
    if ~update_step_size_epoch
      % Update the step_size after every sample
      step_size = step_size_a / (step_size_b + n_samples_drawn)^(step_size_gamma);
    end
    x_batch = x(it_n:min(it_n + batch_size - 1, N_tr), :);
    t_batch = t(it_n:min(it_n + batch_size - 1, N_tr), :);
    N_batch = size(x_batch, 1);
    grad_batch = dpnnErrorGradient(model, x_batch, t_batch, false);
    
    % Implement W <- W - 0.5 * step_size * gradW + randn * sqrt(step_size)
    if strcmp(model.sharing, 'layerwise')
      for l = 1:model.num_layers
        % Gradient for likelihood term must be weighted according to batch size
        grad_batch{l} = grad_batch{l} - model.W{l} / model.gamma(l);
        grad_batch{l} = grad_batch{l} * N_tr / N_batch;
        grad_batch{l} = grad_batch{l} + model.W{l} / model.gamma(l);
        model.W{l} = model.W{l} - 0.5 * step_size * grad_batch{l} + randn(size(grad_batch{l})) * sqrt(step_size);
      end
    elseif strcmp(model.sharing, 'global')
      % Gradient for likelihood term must be weighted according to batch size
      grad_batch = grad_batch - model.W / model.gamma;
      grad_batch = grad_batch * size(x, 1) / batch_size;
      grad_batch = grad_batch + model.W / model.gamma;
      model.W = model.W - 0.5 * step_size * grad_batch + randn(size(grad_batch)) * sqrt(step_size);
    end
    n_samples_drawn = n_samples_drawn + 1;
  end
  
  % Collect sample
  if it > num_samples_burnin
    % Transform weights into vector format
    if strcmp(model.sharing, 'layerwise')
      w_sample = zeros(1, sum(model.num_unique_weights));
      start_idx = 1;
      for l = 1:model.num_layers
        end_idx = start_idx + model.num_unique_weights(l) - 1;
        w_sample(start_idx:end_idx) = model.W{l}(:);
        start_idx = end_idx + 1;
      end
    elseif strcmp(model.sharing, 'global')
      w_sample = model.W;
    else
      error('Error in ''dpnnSampleSGLD'': Unrecognized sharing ''%s''', model.sharing);
    end
    out.saved(it - num_samples_burnin, :) = w_sample;
    if it == num_samples_burnin + 1
      out.squared_distance(1) = 0;
    else
      out.squared_distance(it - num_samples_burnin) = sum((w_sample - out.saved(it - num_samples_burnin - 1, :)).^2);
    end
  end

  if do_output
    debug_f = dpnnErrorFunction(model, x, t);
    debug_err = dpnnEvaluate(model, x, t);
    fprintf('%4d: f=%10.4f, err=%7.5f, step_size=%e\n', it, debug_f, debug_err, step_size);
  end
end

last_model = model;

end

