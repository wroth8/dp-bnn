function [ystar err] = bayesLogRegPred(X, samples, ytrue)

S = size(samples,1);
[D,N] = size(X);
X = [ones(1,N); X]; % append for bias (D+1 x N)

eta = samples*X; % NxS
ystar = sigmoid(eta');

% zero-one error
diffs = abs(double(ystar>0.5) - repmat(ytrue,1,S));
err.zoe = mean(diffs(:));

% test likelihood/cross entropy error
ix0 = (ytrue == 0);
ix1 = ~ix0;
for i = 1:S
    tmpLL(i) = -mean(log2([1-ystar(ix0,i); ystar(ix1,i)]));
end;
err.testLogLik = mean(tmpLL);


