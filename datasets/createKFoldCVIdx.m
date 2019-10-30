function [ folds ] = createKFoldCVIdx( n_samples, n_folds, seed )
% Creates the indices of k-fold cross validation for the given amount of
% samples.
%
% @author Wolfgang Roth

rng(seed);

idx_all = randperm(n_samples);
n_folds_larger = mod(n_samples, n_folds); % number of folds that get assigned one element more
fold_size = floor(n_samples / n_folds); % normal fold size

folds = struct([]);

% Create folds where the training size is one larger
start_idx = 1;
for fold_idx = 1:n_folds_larger
  idx_tr = idx_all;
  idx_tr(start_idx:(start_idx+fold_size)) = [];
  idx_te = idx_all(start_idx:(start_idx+fold_size));
  folds(fold_idx).idx_tr = idx_tr;
  folds(fold_idx).idx_te = idx_te;
  start_idx = start_idx + fold_size + 1;
end

% Process other folds
for fold_idx = (n_folds_larger+1):n_folds
  idx_tr = idx_all;
  idx_tr(start_idx:(start_idx+fold_size-1)) = [];
  idx_te = idx_all(start_idx:(start_idx+fold_size-1));
  folds(fold_idx).idx_tr = idx_tr;
  folds(fold_idx).idx_te = idx_te;
  start_idx = start_idx + fold_size;
end

end

