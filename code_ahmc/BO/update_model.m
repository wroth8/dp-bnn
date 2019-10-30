function model = update_model(model, f_t, final_xatmin, learn_hyper)

model.n = model.n + 1;
model.X(model.n,:) = final_xatmin;
model.f = [model.f', f_t]';


if f_t > model.max_val
	model.max_val = f_t;
    model.scale = 4/model.max_val;
end

model = update_kernel(model, final_xatmin, learn_hyper);

