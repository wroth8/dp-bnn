function [mean, var] = mean_var(model, x)


k_tt = model.cov_model(model.hyp, x, x);
k_x = model.cov_model(model.hyp, model.X(1:model.n,:), x);

mean = k_x'*model.sparse_kernel_inv*model.f*model.scale;
var = diag(k_tt - k_x'*model.sparse_kernel_inv*k_x);
