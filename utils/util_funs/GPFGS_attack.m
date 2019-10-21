function [out,class1,maxminmode] = GPFGS_attack(lp_curr,epsilons_ii,step_size,testPoint,n_steps,params_for_gp_toolbox,bound_comp_opts)
global training_data
global training_labels
if exp(lp_curr) > 0.5 %if mean already above 0.5, need to decrease mean
    perturb = -epsilons_ii;
    perturb_step = -step_size;
    class1 = true; %mark which class test point belongs to
    maxminmode = 'min';
else
    perturb = epsilons_ii;
    perturb_step = step_size;
    class1 = false;
    maxminmode = 'max';
end

jj = 0;
x_adv = testPoint;
while jj < n_steps
    x_adv = GPFGS_step(x_adv,(perturb/n_steps),params_for_gp_toolbox,bound_comp_opts);
    jj = jj+1;
end
[~, ~, ~, ~, lp_adv, ~] = gp(params_for_gp_toolbox.hyp, params_for_gp_toolbox.infFun, params_for_gp_toolbox.meanfunc, ...
    params_for_gp_toolbox.covfunc, params_for_gp_toolbox.likfunc, training_data, training_labels, x_adv, 1);
%attacks_GPFGS.probs(dd+1,ii) = exp(lp_adv);
out = exp(lp_adv);
end