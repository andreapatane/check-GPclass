function [out,class1] = GPJM_attack(testPoint,n_steps,epsilons_ii,lp_curr,params_for_gp_toolbox,bound_comp_opts,step_size)

global training_data
global training_labels

jj = 0;
x_adv = testPoint;
next_step = true;

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


while next_step
    [x_temp,mean_temp] = GPJM_step(x_adv,(perturb/n_steps),params_for_gp_toolbox,bound_comp_opts);
    %Ending conditions: either attack succesfull, of max
    %perturbation reached, or max iterations reached
    if max(abs(x_temp-testPoint)) <= epsilons_ii + 0.00000000001 %needs to be added for numerical imprecision
        x_adv = x_temp;
        if class1 && (mean_temp < 0)
            next_step = false;
        elseif (~class1) && (mean_temp > 0)
            next_step = false;
        end
    else
        disp('x went out of bounds at attack iteration')
        disp(jj)
        %xcomp1(:,ccount) = x_temp;
        %xcomp2(:,ccount) = testPoint;
        %xcomp3(ccount) = epsilons_ii;
        %ccount = ccount +1;
        next_step = false;
    end
    if jj+1 >= n_steps

        next_step = false;
    end
    jj = jj+1;
end
[~, ~, ~, ~, lp_adv2, ~] = gp(params_for_gp_toolbox.hyp, params_for_gp_toolbox.infFun, params_for_gp_toolbox.meanfunc, ...
    params_for_gp_toolbox.covfunc, params_for_gp_toolbox.likfunc, training_data, training_labels, x_adv, 1);
%attacks_GPJM.probs(dd+1,ii) = exp(lp_adv2);
out = exp(lp_adv2);

end