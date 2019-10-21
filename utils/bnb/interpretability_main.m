function [pi_LL,pi_UU,pi_LLm,pi_UUm] = interpretability_main(pix_2_mod_curr,bound_comp_opts,testPoint,testIdx,params_for_gp_toolbox,trainedSystem)


bound_comp_opts.pix_2_mod = pix_2_mod_curr;
[x_L, x_U] = compute_hyper_onesidedinterval(bound_comp_opts.epsilon,testPoint,...
    bound_comp_opts.pix_2_mod,bound_comp_opts.constrain_2_one);
bound_comp_opts.x_L = x_L;
bound_comp_opts.x_U = x_U;
[pi_LL,pi_UU, ~,~,~,~] = main_pi_hat_computation('all',testPoint,testIdx,...
    params_for_gp_toolbox,bound_comp_opts,trainedSystem);
%pi_LLs_a(ii,dd) = pi_LL;
%pi_UUs_a(ii,dd) = pi_UU;


[x_L, x_U] = compute_hyper_onesidedinterval(-bound_comp_opts.epsilon,testPoint,...
    bound_comp_opts.pix_2_mod,bound_comp_opts.constrain_2_one);
bound_comp_opts.x_L = x_U;
bound_comp_opts.x_U = x_L;
[pi_LLm,pi_UUm, ~,~,~,~] = main_pi_hat_computation('all',testPoint,testIdx,...
    params_for_gp_toolbox,bound_comp_opts,trainedSystem);
%pi_LLs_m(ii,dd) = pi_LLm;
%pi_UUs_m(ii,dd) = pi_UUm;

end