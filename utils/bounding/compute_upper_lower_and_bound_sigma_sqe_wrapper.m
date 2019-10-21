function [sigma_l,x_sigma_l,sigma_u,x_sigma_u] = compute_upper_lower_and_bound_sigma_sqe_wrapper(x_L,x_U,theta_vec, ...
    sigma_prior,z_i_L_vec,z_i_U_vec,bound_comp_opts,mu_sigma_bounds,low_or_up,offset_row,offset_cols)
if nargin < 11
    offset_cols = 0;
    if nargin < 10
        offset_row = 0;
        if nargin < 9
            low_or_up = 'both';
        end
    end
end
if strcmp(bound_comp_opts.var_bound,'quick')
    [sigma_l,x_sigma_l,sigma_u,x_sigma_u] = compute_upper_lower_and_bound_sigma_sqe(x_L,x_U,theta_vec,...
        sigma_prior,z_i_L_vec,z_i_U_vec,bound_comp_opts,mu_sigma_bounds,low_or_up,offset_row,offset_cols);
elseif strcmp(bound_comp_opts.var_bound,'slow')
    [sigma_l,x_sigma_l,sigma_u,x_sigma_u] = compute_upper_lower_and_bound_sigma_sqe_quad_prog(x_L,x_U,theta_vec,...
        sigma_prior,z_i_L_vec,z_i_U_vec,bound_comp_opts,mu_sigma_bounds,low_or_up);
end
    
                        
                        
end