% MNIST Basic
% Download mnist.zip from
% - https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits
% - http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip
% Unpack the zipfile and run this script to get the 50-dimensional PCA
% version.
% Note: This is NOT the original MNIST dataset. Here the split into
% training, test and validation set is different.
% Note: The same code can be used to create the PCA-50 versions of the
% remaining MNIST variants which can be found on the same website as MNIST
% Basic. You just have to replace the 'load(...)' operation accordingly.
%
% @author Wolfgang Roth

close all;
clear;

x_tr = load('mnist_train.amat');
x_te = load('mnist_test.amat');

% Divide training data into training data and validation data (10000 train, 2000 validation)
x_va = x_tr(10001:end, :);
x_tr = x_tr(1:10000, :);

% Extract classes from the last column
t_tr = x_tr(:, end);
t_va = x_va(:, end);
t_te = x_te(:, end);
x_tr = x_tr(:, 1:end-1);
x_va = x_va(:, 1:end-1);
x_te = x_te(:, 1:end-1);
t_tr(t_tr == 0) = 10;
t_va(t_va == 0) = 10;
t_te(t_te == 0) = 10;

% Create PCA reduced dataset
coeff = pca(x_tr);
mean_data = mean(x_tr);
pca_D = 50;
x_tr = bsxfun(@minus, x_tr, mean_data) * coeff;
x_tr = x_tr(:, 1:pca_D);
x_va = bsxfun(@minus, x_va, mean_data) * coeff;
x_va = x_va(:, 1:pca_D);
x_te = bsxfun(@minus, x_te, mean_data) * coeff;
x_te = x_te(:, 1:pca_D);

% Normalize PCA reduced set to have unit variance in each dimension (whitening)
std_data = std(x_tr);
x_tr = bsxfun(@times, x_tr, 1 ./ std_data);
x_va = bsxfun(@times, x_va, 1 ./ std_data);
x_te = bsxfun(@times, x_te, 1 ./ std_data);
save(sprintf('mnist_basic_pca%d_zeromean.mat', pca_D), 'x_tr', 't_tr', 'x_va', 't_va', 'x_te', 't_te');
