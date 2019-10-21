%function [lbstar,ubstar,x_over_star] = compute_curr_lb_ub(max_or_min,mode,...
%    kernel_name,x_L,x_U,training_data,training_labels,params_for_gp_toolbox,bound_comp_opts,varargin)
function [lbstar,ubstar,x_over_star,mu_sigma_bounds,bound_comp_opts] = compute_curr_lb_ub(testIdx,max_or_min,mode,...
    kernel_name,x_L,x_U,params_for_gp_toolbox,bound_comp_opts,x_over,mu_sigma_bounds,varargin)


switch mode
    case 'binarypi'
        trainedSystem = varargin{1};
        [lbstar,ubstar,x_over_star,mu_sigma_bounds,bound_comp_opts] = compute_lb_ub_binarypi(testIdx,max_or_min,...
                kernel_name,x_L,x_U,params_for_gp_toolbox,bound_comp_opts,trainedSystem,mu_sigma_bounds,x_over);
    case 'binarymu'
        error('still need to implement this')
    case 'multipi'
        trainedSystem = varargin{1};
        [lbstar,ubstar,x_over_star,mu_sigma_bounds,bound_comp_opts] = compute_lb_ub_multipi(testIdx,max_or_min,...
                kernel_name,x_L,x_U,params_for_gp_toolbox,bound_comp_opts,trainedSystem,mu_sigma_bounds,x_over);


end


end