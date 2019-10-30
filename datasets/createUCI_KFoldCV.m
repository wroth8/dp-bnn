% Creates the k-fold cross validation splits for all UCI datasets.
%
% @author Wolfgang Roth

createKFoldCVIdxSeparateFiles(506, 5, 1, 'folds/housing');
createKFoldCVIdxSeparateFiles(4177, 5, 1, 'folds/abalone');
createKFoldCVIdxSeparateFiles(1030, 5, 1, 'folds/concrete');
createKFoldCVIdxSeparateFiles(9568, 5, 1, 'folds/powerplant');
createKFoldCVIdxSeparateFiles(1599, 5, 1, 'folds/wineqred');
createKFoldCVIdxSeparateFiles(4898, 5, 1, 'folds/wineqwhite');
