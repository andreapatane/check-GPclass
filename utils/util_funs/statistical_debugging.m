function [ pi_L_hat,  pi_U_hat ] = statistical_debugging(bound_comp_opts,testPoint,params_for_gp_toolbox,N)
global post
global training_data
if isfield(bound_comp_opts,'x_L')
    x_L = bound_comp_opts.x_L;
    x_U = bound_comp_opts.x_U;    
else
    [x_L, x_U] = compute_hyper_rectangle(bound_comp_opts.epsilon,testPoint,...
        bound_comp_opts.pix_2_mod,bound_comp_opts.constrain_2_one);    
end


pix2mod = find(x_L < x_U);

pi_L_hat =  inf;
pi_U_hat = - inf;


for ii = 1:N
    x_curr = x_L;
    rand_unif = rand(length(pix2mod),1);
    for jj = 1:length(pix2mod)
        x_curr(pix2mod(jj)) = ( x_U(pix2mod(jj)) - x_L(pix2mod(jj)) ) * rand_unif(jj) +  x_L(pix2mod(jj));
    end
    [~, ~, ~, ~,lp] = gp(params_for_gp_toolbox.hyp, params_for_gp_toolbox.infFun, ...
                  params_for_gp_toolbox.meanfunc, params_for_gp_toolbox.covfunc, params_for_gp_toolbox.likfunc,...
                  training_data, post, x_curr, ones(1,1));
    prob_out = exp(lp);
    if prob_out < pi_L_hat
        pi_L_hat = prob_out;
    end
    if prob_out > pi_U_hat
        pi_U_hat = prob_out;
    end
end



end