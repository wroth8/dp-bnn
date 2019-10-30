function model_nn = dpnn2nn( model_dpnn )
% Converts the given DP neural network to a plain feed forward neural
% network with the same weights. Note: The seeds from model_dpnn will be
% copied to model_nn, but the random number generators are not changed.
% In case of layerwise sharing, only the first entry of gamma and
% init_gamma are copied.
%
% Input:
% model_dpnn: The DP neural network (see dpnnInitCRP)
%
% Output:
% model_nn: The plain feed-forward neural network (see nnInit)
%
% @author Wolfgang Roth

W = cell(1, model_dpnn.num_layers);
b = cell(1, model_dpnn.num_layers);
if strcmp(model_dpnn.sharing, 'global')
  for l = 1:model_dpnn.num_layers
    W{l} = reshape(model_dpnn.W(model_dpnn.ZW{l}), size(model_dpnn.ZW{l}));
    b{l} = reshape(model_dpnn.W(model_dpnn.Zb{l}), size(model_dpnn.Zb{l}));
  end
elseif strcmp(model_dpnn.sharing, 'layerwise')
  for l = 1:model_dpnn.num_layers
    W{l} = reshape(model_dpnn.W{l}(model_dpnn.ZW{l}), size(model_dpnn.ZW{l}));
    b{l} = reshape(model_dpnn.W{l}(model_dpnn.Zb{l}), size(model_dpnn.Zb{l}));
  end
else
  error('Error in ''dpnn2nn'': Unrecognized sharing ''%s''', sharing);
end

num_unique_weights = sum(cellfun(@(w) numel(w), W)) + sum(model_dpnn.layout(2:end));

model_nn = struct('layout',              model_dpnn.layout,        ...
                  'task',                model_dpnn.task,          ...
                  'activation',          model_dpnn.activation,    ...
                  'num_layers',          model_dpnn.num_layers,    ...
                  'W',                   {W},                      ...
                  'b',                   {b},                      ...
                  'num_unique_weights',  num_unique_weights,       ...
                  'beta',                model_dpnn.beta,          ...
                  'gamma',               model_dpnn.gamma(1),      ...
                  'init_gamma',          model_dpnn.init_gamma(1), ...
                  'rng_init_seed',       model_dpnn.rng_init_seed_matlab);


end
