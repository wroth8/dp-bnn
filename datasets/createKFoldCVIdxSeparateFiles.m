function [ folds ] = createKFoldCVIdxSeparateFiles( n_samples, n_folds, seed, file_prefix )
% Creates the indices of k-fold cross validation for the given amount of
% samples.
%
% @author Wolfgang Roth

folds = createKFoldCVIdx(n_samples, n_folds, seed);

for fold_idx = 1:n_folds
  idx_tr = folds(fold_idx).idx_tr;
  idx_te = folds(fold_idx).idx_te;
  save(sprintf('%s_fold%d.mat', file_prefix, fold_idx), 'idx_tr', 'idx_te');
end

end

