function model = dpnnInitUniform( layout, max_num_weights, task, sharing, activation, alpha, beta, gamma, init_gamma, rng_seed )
% Samples a Dirichlet process (DP) neural network. The sharing
% configuration is initialized uniformely, i.e. the given amount of maximum
% weights are equally distributed among the connections. In other words,
% each weight is used equally often (weight usages differ by at most one).
% The weights are sampled from a zero mean Gaussian with variance
% init_gamma.
%
% Input:
% layout: Row vector determining the structure of the neural network. The
%   first entry determines the number of input units. The last entry
%   determines the number of output units. The intermediate entries define
%   the number of neurons per hidden layer (and implicitly the number of
%   hidden layers).
% max_num_weights: Determines the number of weights. In case of 'layerwise'
%   sharing, max_num_weights must be a vector containing the number of
%   different weights per layer. In case of 'global' sharing, 
%   max_num_weights must be a scalar value determining the overall amount
%   of weights. If max_num_weights is larger than the number of actual
%   connections, every connection will get a different weight.
% task: Determines the task which should be solved with the network:
%   'regress': There are real valued outputs
%   'biclass': There are (possibly multiple) binary outputs
%   'muclass': There are multiple binary outputs where exactly one is true
% sharing: Determines how the weights should be shared:
%   'layerwise': The weights are only shared within a layer
%   'global': The weights are shared among the layers
% activation: Determines the activation functions of the neurons:
%   'sigmoid': The logistic sigmoid function 1/(1+exp(-x))
%   'tanh': The tangens hyperbolicus function (exp(x)-exp(-x))/(exp(x)+exp(-x))
%   'relu': The rectifier linear unit max(0,x)
% alpha: The concentration parameter of the Dirichlet process. In case of
%   'layerwise' sharing the parameter can be given as a row vector
%   containing the alpha parameter for each layer or as a scalar in which
%   case the same alpha will be used for all layers. In case of 'global'
%   sharing alpha must be a scalar. This parameter is not used to
%   initialize the sharing configuration.
% beta: Determines the confidence of the output. In case of 'regress' this
%   value determines the variance on the target values. In case of
%   'biclass' the output activation is sigmoid(beta*x). In case of
%   'muclass' the output activation is softmax(beta*x). This parameter can
%   be used to control the influence of the likelihood.
% gamma: The variance of the base function of the Dirichlet process
%   (basically the variance of the weights). In case of 'layerwise' sharing
%   the parameter can be given as a row vector containing the gamma
%   parameter for each layer or as a scalar in which case the same gamma
%   will be used for all layers. In case of 'global' sharing alpha must be
%   a scalar. This parameter is not used to initialize the weights.
% init_gamma: The variance of the zero mean Gaussian that is used to
%   initialize the weights. This can be different from 'gamma' because the
%   performance of several algorithms depend heavily on the initialization
%   of the weights.
% rng_seed: The seed used to initialize the Matlab and the C++ random
%   number generator. If this variable is set to a negative value, the
%   random number generator will be initialized using rng('shuffle').
%   Otherwise the random number generator will be initialized using the
%   given seed.
%
% Output:
% model: Struct containing the sampled Dirichlet process neural network
%   'layout': see inputs
%   'task': see inputs
%   'sharing': see inputs
%   'activation': see inputs
%   'num_layers': The number of layers of the neural network. This is
%      counted as the number of weight matrices (see Bishop PRML
%      terminology).
%   'ZW': Cell array containing the weight indicators of each layer.
%   'Zb': Cell array containing the weight indicators for the biases of
%     each layer.
%   'W': In case of global sharing a row vector containing the weights of
%     the network. In case of layerwise sharing a cell array containing row
%     vectors of the weights of each layer.
%   'num_weights': In case of global sharing a row vector containing the
%     number of how often a particular weight is being shared. In case of
%     layerwise sharing a cell array containing row vectors with the number
%     of how often a particular weight is being shared within its layer.
%   'num_unique_weights': In case of global sharing a row vector containing
%     the number of different weights in the whole network. In case of
%     layerwise sharing the number of different weights within each layer.
%   'alpha': see inputs (in case of 'layerwise' sharing this parameter will
%     always be a vector regardless of whether the parameter was given as a
%     scalar or a vector)
%   'beta': see inputs
%   'gamma': see inputs (in case of 'layerwise' sharing this parameter will
%     always be a vector regardless of whether the parameter was given as a
%     scalar or a vector)
%   'init_gamma': see inputs (in case of 'layerwise' sharing this parameter
%     will always be a vector regardless of whether the parameter was given
%     as a scalar or a vector)
%   'rng_init_seed_matlab': The seed that was used to initialize the Matlab
%     random number generator. The seed is stored for reproduction of the
%     results.
%   'rng_init_seed_native': The seed that was used to initialize the C++
%     random number generator. The seed is stored for reproduction of the
%     results.
%
% @author Wolfgang Roth

if ~isrow(layout)
  error('Error in ''dpnnInitUniform'': Argument ''layout'' must be a row vector');
end
layout = int32(layout);

num_layers = int32(length(layout) - 1);

if ~strcmp(task, 'regress') && ~strcmp(task, 'biclass') && ~strcmp(task, 'muclass')
  error('Error in ''dpnnInitUniform'': Unrecognized task ''%s''', task);
end

if ~strcmp(sharing, 'layerwise') && ~strcmp(sharing, 'global')
  error('Error in ''dpnnInitUniform'': Unrecognized sharing ''%s''', sharing);
end

if ~strcmp(activation, 'sigmoid') && ~strcmp(activation, 'tanh') && ~strcmp(activation, 'relu')
  error('Error in ''dpnnInitUniform'': Unrecognized activation ''%s''', activation);
end

if strcmp(sharing, 'global')
  if ~isscalar(max_num_weights)
    error('Error in ''dpnnInitUniform'': Argument ''max_num_weights'' must be a scalar');
  end
else % strcmp(sharing, 'layerwise')
  if ~isrow(max_num_weights) || length(max_num_weights) ~= num_layers
    error('Error in ''dpnnInitUniform'': Argument ''max_num_weights'' must be a row vector of length ''num_layers''');
  end
end

if strcmp(sharing, 'global')
  if ~isscalar(alpha) || alpha <= 0
    error('Error in ''dpnnInitUniform'':  Argument ''alpha'' must be a positive scalar');
  end
else % strcmp(sharing, 'layerwise')
  if isscalar(alpha)
    alpha = ones(1, num_layers) * alpha;
  end
  if ~isrow(alpha) || any(alpha <= 0) || length(alpha) ~= num_layers
    error('Error in ''dpnnInitUniform'': Argument ''alpha'' must be a row vector with num_layers positive entries or a positive scalar');
  end
end

if ~isscalar(beta) || beta <= 0
  error('Error in ''dpnnInitUniform'': Argument ''beta'' must be a postive scalar');
end

if strcmp(sharing, 'global')
  if ~isscalar(gamma) || gamma <= 0
    error('Error in ''dpnnInitUniform'': Argument ''gamma'' must be a positive scalar');
  end
else % strcmp(sharing, 'layerwise')
  if isscalar(gamma)
    gamma = ones(1, num_layers) * gamma;
  end
  if ~isrow(gamma) || any(gamma <= 0) || length(gamma) ~= num_layers
    error('Error in ''dpnnInitUniform'': Argument ''gamma'' must be a row vector with num_layers positive entries or a positive scalar');
  end
end

if strcmp(sharing, 'global')
  if ~isscalar(init_gamma) || init_gamma <= 0
    error('Error in ''dpnnInitUniform'': Argument ''init_gamma'' must be a positive scalar');
  end
else % strcmp(sharing, 'layerwise')
  if isscalar(init_gamma)
    init_gamma = ones(1, num_layers) * init_gamma;
  end
  if ~isrow(init_gamma) || any(init_gamma <= 0) || length(init_gamma) ~= num_layers
    error('Error in ''dpnnInitUniform'': Argument ''init_gamma'' must be a row vector with num_layers positive entries or a positive scalar');
  end
end

if ~isscalar(rng_seed)
  error('Error in ''dpnnInitUniform'': Argument ''rng_seed'' must be a scalar'); 
end

if rng_seed < 0
  rng('shuffle', 'twister');
else
  rng(rng_seed, 'twister');
end
rng_seed_matlab = rng;
mexRngInit(rng_seed_matlab.Seed);
rng_seed_native = mexRngState;

ZW = cell(1, num_layers);
Zb = cell(1, num_layers);

if strcmp(sharing, 'layerwise')
  W = cell(1, num_layers);
  num_weights = cell(1, num_layers);
  num_unique_weights = zeros(1, num_layers, 'int32');
  % Sample each layer separately
  for l = 1:num_layers
    n_weights = (layout(l) + 1) * layout(l+1);
    if max_num_weights(l) > n_weights
      max_num_weights(l) = n_weights;
    end
    z = int32(1:max_num_weights(l));
    z = repmat(z, [1, ceil(double(n_weights) / double(max_num_weights(l)))]);
    z = z(1:n_weights);
    z = z(randperm(n_weights));
    w_idx = layout(l) * layout(l+1);
    ZW{l} = reshape(z(1:w_idx), layout(l), layout(l+1));
    Zb{l} = reshape(z((w_idx+1):end), 1, layout(l+1));
    
    num_weights{l} = int32(accumarray(z', 1, [max_num_weights(l), 1])');
    num_unique_weights(l) = max_num_weights(l);
    W{l} = randn(1, max_num_weights(l)) * sqrt(init_gamma(l));
  end
else % strcmp(sharing, 'global')
  n_weights = 0;
  for l = 1:num_layers
    n_weights = n_weights + (layout(l) + 1) * layout(l+1);
  end
  if max_num_weights > n_weights
    max_num_weights = n_weights;
  end
  z = int32(1:max_num_weights);
  z = repmat(z, [1, ceil(double(n_weights) / double(max_num_weights))]);
  z = z(1:n_weights);
  z = z(randperm(n_weights));
  
  start_idx = 1;
  for l = 1:num_layers
    end_idx = start_idx + layout(l) * layout(l+1) - 1;
    ZW{l} = reshape(z(start_idx:end_idx), layout(l), layout(l+1));
    
    start_idx = end_idx + 1;
    end_idx = start_idx + layout(l+1) - 1;
    Zb{l} = reshape(z(start_idx:end_idx), 1, layout(l+1));
    start_idx = end_idx + 1;
  end
  
  num_weights = int32(accumarray(z', 1, [max_num_weights, 1])');
  num_unique_weights = int32(max_num_weights);
  W = randn(1, max_num_weights) * sqrt(init_gamma);
end

model = struct('layout',               layout,             ...
               'task',                 task,               ...
               'sharing',              sharing,            ...
               'activation',           activation,         ...
               'num_layers',           num_layers,         ...
               'ZW',                   {ZW},               ...
               'Zb',                   {Zb},               ...
               'W',                    {W},                ...
               'num_weights',          {num_weights},      ...
               'num_unique_weights',   num_unique_weights, ...
               'alpha',                alpha,              ...
               'beta',                 beta,               ...
               'gamma',                gamma,              ...
               'init_gamma',           init_gamma,         ...
               'rng_init_seed_matlab', rng_seed_matlab,    ...
               'rng_init_seed_native', rng_seed_native);

end

