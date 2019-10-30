function out = samplingHMC( w, func, grad, num_hmc_iter, leapfrog_L, leapfrog_epsilon, varargin )
% Performs hybrid Monte Carlo for the given log-density and the gradient
% thereof.
%
% Input:
% w: The initial value for hybrid Monte Carlo
% func: Function handle of the negative log-density to sample from
% grad: Function handle of the gradient of the negative log-density
% num_hmc_iter: The number of hybrid Monte Carlo iterations to perform. One
%   iteration is defined as a leapfrog integration with leapfrog_L steps
%   with step size leapfrog_epsilon.
% leapfrog_L: The number of discrete steps of the leapfrog integration.
% leapfrog_epsilon: The step size of the leapfrog integration.
% do_output [optional]: Determines whether some output about the progress
%   of the algorithm should be displayed or not (true or false, default:
%   true).
%
% Output:
% out: Struct containing information about the hybrid Monte Carlo algorithm
%   'saved': Matrix containing the intermediate sampled values of the
%     hybrid Monte Carlo algorithm. Each row corresponds to a drawn sample.
%     Note that the first row contains the initial values.
%   'rejections: Row vector containing logicals indicating if the sample of
%     the corresponding iteration was rejected.
%   'squared_distance': Row vector containing the squared distance of
%     consecutive samples.
%   'accept_prob': Row vector containing the acceptance probability of the
%     corresponding sample.
%
% @author Wolfgang Roth

% Check optional arguments
if length(varargin) > 1
  error('Error in ''samplingHMC'': Too much arguments given');
elseif length(varargin) == 1
  do_output = varargin{1};
else
  do_output = true;
end

out.saved = w;
out.rejection = zeros(1, num_hmc_iter);
out.squared_distance = zeros(1, num_hmc_iter);
out.accept_prob = zeros(1, num_hmc_iter);

for it = 1:num_hmc_iter
  r = randn(size(w));
  H = func(w) + 0.5 * sum(r(:).^2);
  
  grad_w = grad(w);
  r = r - leapfrog_epsilon * 0.5 * grad_w;
  for lf_it = 1:leapfrog_L
    if any(isnan(w)) || any(isnan(r))
      continue;
    end

    w = w + leapfrog_epsilon * r;
    if lf_it ~= leapfrog_L
      grad_w = grad(w);
      r = r - leapfrog_epsilon * grad_w;
    end
  end

  if ~any(isnan(w)) && any(isnan(r))
    grad_w = grad(w);
    r = r - leapfrog_epsilon * 0.5 * grad_w;
  end
  
  if any(isnan(w)) || any(isnan(r))
    warning('Warning in ''samplingHMC'': NaNs occurred during execution. Maybe leapfrog_epsilon is too large?');
    H_new = inf;
  else
    H_new = func(w) + 0.5 * sum(r(:).^2);
  end

  out.accept_prob(it) = min(1, exp(H - H_new));
  if rand < out.accept_prob(it)
    % Sample accepted
    out.rejection(it) = false;
    out.squared_distance(it) = sum((w - out.saved(end, :)).^2);
    if do_output
      fprintf('%4d/%d: Sample accepted: H=%f, H_new=%f\n', it, num_hmc_iter, H, H_new);
    end
  else
    % Sample rejected
    w = out.saved(end, :);
    out.rejection(it) = true;
    out.squared_distance(it) = 0;
    if do_output
      fprintf('%4d/%d: Sample rejected: H=%f, H_new=%f\n', it, num_hmc_iter, H, H_new);
    end
  end
  out.saved = [out.saved; w];
end

end

