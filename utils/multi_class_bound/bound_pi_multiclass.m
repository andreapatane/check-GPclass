function pi_out = bound_pi_multiclass(mu_l,mu_u,sigma_l,sigma_u,mode)
global multi_grid_extreme multi_grid
pi_out = 0;

[sigma_f_u,sigma_f_l] = comput_sigma_fs(sigma_l,sigma_u);
for ii = 1:length(multi_grid)
    %mu_f computations
    [mu_f_u,mu_f_l] = comput_mu_fs(mu_l,mu_u,sigma_l,sigma_u,multi_grid{ii});
    %sigma_f computation
    %1-d bound computations and merging
    pi_out = pi_out + multi_grid_extreme(ii)*merging_1d_bounds(mu_f_u,mu_f_l,sigma_f_u,sigma_f_l,multi_grid{ii},mode);
    %pi_out = pi_out + multi_grid_extreme(ii)*bound_multinormal_integral(mu_l,mu_u,sigma_l,sigma_u,multi_grid{ii},mode);   
end




end