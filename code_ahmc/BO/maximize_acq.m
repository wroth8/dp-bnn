function x_max = maximize_acq(model, dopt, type, si) 
options = [];
if nargin < 3
	type = 'ucb';
end

if nargin < 4
	si = 1;
end

problem.f = @(x) acq(model, x', type, si);	

funProj = @(x) max(min(x, model.bounds(:, 2)),  model.bounds(:, 1));

if model.use_direct
	[ret_minval, x_max, history] = direct(problem, model.bounds, dopt);
	if ~isdeployed
		options.verbose = 0;
		options.maxIter = 100;
		[x_max, f] = minConf_TMP(problem.f, x_max, model.bounds(:, 1), model.bounds(:, 2), options);
	end
else
	options.verbose = 0;
	options.maxIter = 100;

	num = min(model.n, ceil(2*sqrt(model.n)));

	max_f = 1e16;
	x_max = 0;
	
  	% Random starting pts based on sampled pts
	indices = randperm(model.n);
	indices = indices(1:num);
	for i = 1:size(indices)
		init_pt = model.X(indices(i),:)'+randn(model.d,1)*1e-3;
		[x_local, f] = minConf_SPG(problem.f, init_pt, funProj, options);
		
		if f < max_f
			max_f = f;
			x_max = x_local;
		end	
	end
	
	old_max_f = max_f;
	old_x_max = x_max;
	
	
	num = 5*model.d;
	
	% Random starting pts by using Latin Hyper Cube
	lh_pts = lhsu(model.bounds(:, 1),model.bounds(:, 2),num);
	for i = 1:size(lh_pts, 1)
		[x_local, f] = minConf_Tmp(problem.f, lh_pts(i,:)', model.bounds(:, 1), model.bounds(:, 2), options);
		if f < max_f
			max_f = f;
			x_max = x_local;
		end	
	end
	
	
%  	if max_f < old_max_f
%  		disp('NO NO NO NO NO NO NO!')
%  		disp([old_x_max', -old_max_f])
%  		disp([x_max', -max_f])
%  		
%  	end
	
end


