function model = init_model(d, bounds, init_pt, init_f, hyp, noise, ...
	cov_model)

model.use_direct = 1;
model.noise = noise;

if strcmp(cov_model,'se')
	model.cov_model = @(hyp, x, z, records)covSEiso(hyp, x, z);
elseif strcmp(cov_model,'ard')
	model.cov_model = @(hyp, x, z, records)covSEard(hyp, x, z);
end

model.hyp = hyp;
model.bounds = bounds;
model.d = d;


model.scale = 1;
model.sparse_kernel_inv = 1/(model.cov_model(model.hyp, init_pt, init_pt) ...
	+ model.noise);
model.X = zeros(3000, d);
model.X(1, :) = init_pt;

model.f = init_f;
model.m = 1;
model.n = 1;

model.max_val = init_f;
model.display = 1;
