function [out, last_model] = dpnnSampleWeightsHMC( model, x, t, num_hmc_iter, leapfrog_L, leapfrog_epsilon, varargin )
% Performs the hybrid Monte Carlo algorithm using samplingHMC to sample the
% weights of DP neural networks.
%
% Input:
% model: The initial DP neural network (see dpnnInitCRP)
% x: The input vectors stored in a matrix where rows correspond to
%   different samples and columns correspond to different features.
% t: The output vectors stored in a matrix where rows correspond to
%   different samples. In case the task of the model is 'regress' or
%   'biclass', the columns correspond to different outputs. In case the
%   task of the model is 'muclass', t must be a column vector containing
%   the class labels as integers starting from 1.
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
%     Note, that the initial values are not stored.
%   'rejections: Row vector containing logicals indicating if the sample of
%     the corresponding iteration was rejected.
%   'squared_distance': Row vector containing the squared distance of
%     consecutive samples.
%   'accept_prob': Row vector containing the acceptance probability of the
%     corresponding sample.
% last_model: The last sampled model as DP neural network struct (see
%   dpnnInitCRP).
%
% Note: x and t can be given as gpuarray to utilize GPU computation
%
% @author Wolfgang Roth

% Check optional arguments
if length(varargin) > 1
  error('Error in ''dpnnSampleWeightsHMC'': Too much arguments given');
elseif length(varargin) == 1
  do_output = varargin{1};
else
  do_output = true;
end

% Transform weights into vector format
if strcmp(model.sharing, 'layerwise')
  w0 = zeros(1, sum(model.num_unique_weights));
  start_idx = 1;
  for l = 1:model.num_layers
    end_idx = start_idx + model.num_unique_weights(l) - 1;
    w0(start_idx:end_idx) = model.W{l}(:);
    start_idx = end_idx + 1;
  end
elseif strcmp(model.sharing, 'global')
  w0 = model.W;
else
  error('Error in ''dpnnSampleWeightsHMC'': Unrecognized sharing ''%s''', model.sharing);
end

% Perform hybrid Monte Carlo
func = @(w) funDpnnErrorFunction(w, model, x, t);
grad = @(w) funDpnnErrorGradient(w, model, x, t);
out = samplingHMC(w0, func, grad, num_hmc_iter, leapfrog_L, leapfrog_epsilon, do_output);

% Transform last model back
if strcmp(model.sharing, 'layerwise')
  start_idx = 1;
  for l = 1:model.num_layers
    end_idx = start_idx + model.num_unique_weights(l) - 1;
    model.W{l} = out.saved(end, start_idx:end_idx);
    start_idx = end_idx + 1;
  end
elseif strcmp(model.sharing, 'global')
  model.W = out.saved(end, :);
else
  error('Error in ''dpnnSampleWeightsHMC'': Unrecognized sharing ''%s''', model.sharing);
end
last_model = model;

end

