function model = nnInit( layout, task, activation, beta, gamma, init_gamma, rng_seed )
% Samples a plain feed-forward neural network. The weights are sampled from
% a zero mean Gaussian with variance init_gamma.
%
% Input:
% layout: Determines the structure of the neural network. The first entry
%   determines the number of input units. The last entry determines the
%   number of output units. The intermediate entries define the number of
%   neurons per hidden layer (and implicitly the number of hidden layers).
% task: Determines the task which should be solved with the network:
%   'regress': There are real valued outputs
%   'biclass': There are (possibly multiple) binary outputs
%   'muclass': There are multiple binary outputs where exactly one is true
% activation: Determines the activation functions of the neurons:
%   'sigmoid': The logistic sigmoid function 1/(1+exp(-x))
%   'tanh': The tangens hyperbolicus function (exp(x)-exp(-x))/(exp(x)+exp(-x))
%   'relu': The rectifier linear unit max(0,x)
% beta: Determines the confidence of the output. In case of 'regress' this
%   value determines the variance on the target values. In case of
%   'biclass' the output activation is sigmoid(beta*x). In case of
%   'muclass' the output activation is softmax(beta*x). This parameter can
%   be used to control the influence of the likelihood.
% gamma: The variance of the zero mean Gaussian prior over the weights.
%   This parameter is not used to initialize the weights.
% init_gamma: The variance of the zero mean Gaussian that is used to
%   initialize the weights. This can be different from 'gamma' because the
%   performance of several algorithms depend heavily on the initialization
%   of the weights.
% rng_seed: The seed used to initialize the Matlab random number generator.
%   If this variable is set to a negative value, the random number
%   generator will be initialized using rng('shuffle'). Otherwise the
%   random number generator will be initialized using the given seed.
%
% Output:
% model: Struct containing the sampled feed-forward neural network
%   'layout': see inputs
%   'task': see inputs
%   'sharing': see inputs
%   'activation': see inputs
%   'num_layers': The number of layers of the neural network. This is
%      counted as the number of weight matrices (see Bishop PRML
%      terminology).
%   'W': Cell array containing the weight matrices of each layer.
%   'b': Cell array containing the biases of each layer.
%   'num_unique_weights': The number of weights and biases of the neural
%     network.
%   'beta': see inputs
%   'gamma': see inputs
%   'init_gamma': see inputs
%   'rng_init_seed': The seed that was used to initialize the Matlab random
%     number generator. The seed is stored for reproduction of the results.
%
% Note: Only the RNG for Matlab is set here unlike
%   dpnnInitCRP/dpnnInitUniform function where also the native RNG is
%   initialized.
%
% @author Wolfgang Roth

if ~isrow(layout)
  error('Error in ''nnInit'': Argument ''layout'' must be a row vector');
end
layout = int32(layout);

num_layers = int32(length(layout) - 1);

if ~strcmp(task, 'regress') && ~strcmp(task, 'biclass') && ~strcmp(task, 'muclass')
  error('Error in ''nnInit'': Unrecognized task ''%s''', task);
end

if ~strcmp(activation, 'sigmoid') && ~strcmp(activation, 'tanh') && ~strcmp(activation, 'relu')
  error('Error in ''nnInit'': Unrecognized activation ''%s''', activation);
end

if ~isscalar(beta) || beta <= 0
  error('Error in ''nnInit'': Argument ''beta'' must be a postive scalar');
end
  
if ~isscalar(gamma) || gamma <= 0
  error('Error in ''nnInit'': Argument ''gamma'' must be a positive scalar');
end

if ~isscalar(init_gamma) || init_gamma <= 0
  error('Error in ''nnInit'': Argument ''init_gamma'' must be a positive scalar');
end

if ~isscalar(rng_seed)
  error('Error in ''nnInit'': Argument ''rng_seed'' must be a scalar'); 
end

if rng_seed < 0
  rng('shuffle', 'twister');
else
  rng(rng_seed, 'twister');
end
rng_seed = rng;

W = cell(1, num_layers);
b = cell(1, num_layers);
for l = 1:num_layers
  W{l} = randn(layout(l), layout(l+1)) * sqrt(init_gamma);
  b{l} = randn(1, layout(l+1)) * sqrt(init_gamma);
end
num_unique_weights = sum(cellfun(@(w) numel(w), W)) + sum(layout(2:end));

model = struct('layout',              layout,             ...
               'task',                task,               ...
               'activation',          activation,         ...
               'num_layers',          num_layers,         ...
               'W',                   {W},                ...
               'b',                   {b},                ...
               'num_unique_weights',  num_unique_weights, ...
               'beta',                beta,               ...
               'gamma',               gamma,              ...
               'init_gamma',          init_gamma,         ...
               'rng_init_seed',       rng_seed);

end

