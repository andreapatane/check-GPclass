%function [out_l,out_u,x_over_star] = compute_lb_ub_binarypi(max_or_min,...
%    kernel_name,x_L,x_U,training_data,training_labels,params_for_gp_toolbox,bound_comp_opts,trainedSystem,S)
function [out_l,out_u,x_over_star,mu_sigma_bounds,bound_comp_opts] = compute_lb_ub_binarypi(testIdx,max_or_min,...
    kernel_name,x_L,x_U,params_for_gp_toolbox,bound_comp_opts,trainedSystem,mu_sigma_bounds,x_over)


%global training_labels
global training_data
global post
global pred_var

global std_time
global mu_time
global discrete_time
global inference_time

%Minimum Case:

%1 - I compute the minimum latent min value


switch bound_comp_opts.likmode
    case 'analytical'
        %Case in which the integral can be solved analytically for minimum
        %and maximum values.
        if isequal(params_for_gp_toolbox.likfunc,@likErf)

            
            switch kernel_name
  
                case 'sqe'
                    [z_i_L_vec,z_i_U_vec] = pre_compute_z_intervals(x_L,x_U,params_for_gp_toolbox.theta_vec);
                    %1 - first I compute the optimal mean value
                    if strcmp(max_or_min,'min')
                        %[mu_star,x_mu_star] = compute_lower_bound_mu_sqe(trainedSystem,x_L,x_U,params_for_gp_toolbox.theta_vec,...
                        %    params_for_gp_toolbox.sigma,training_data);
                        [mu_star,x_mu_star] = compute_lower_bound_mu_sqe(trainedSystem,x_L,x_U,params_for_gp_toolbox.theta_vec,...
                            params_for_gp_toolbox.sigma,z_i_L_vec,z_i_U_vec);
                        mu_flag = mu_star;
                    else
                        %[mu_star,x_mu_star] = compute_upper_bound_mu_sqe(trainedSystem,x_L,x_U,params_for_gp_toolbox.theta_vec,...
                        %    params_for_gp_toolbox.sigma,training_data);
                         [mu_star,x_mu_star] = compute_upper_bound_mu_sqe(trainedSystem,x_L,x_U,params_for_gp_toolbox.theta_vec,...
                            params_for_gp_toolbox.sigma,z_i_L_vec,z_i_U_vec);
                        mu_flag =  - mu_star;
                    end
                    %2 - I then compute the optimal variance value depending on
                    %the value obtained for the mean
                    if mu_flag >= 0
                        %[sigma_star,x_sigma_star] = compute_upper_bound_sigma_sqe(x_L,x_U,params_for_gp_toolbox.theta_vec, ...
                        %    params_for_gp_toolbox.sigma,z_i_L_vec,z_i_U_vec);
                        
                        %[~,~,sigma_star,x_sigma_star] = compute_upper_lower_and_bound_sigma_sqe_quad_prog(x_L,x_U,params_for_gp_toolbox.theta_vec, ...
                        %    params_for_gp_toolbox.sigma,z_i_L_vec,z_i_U_vec,bound_comp_opts,mu_sigma_bounds,'upper');
                        [~,~,sigma_star,x_sigma_star] = compute_upper_lower_and_bound_sigma_sqe_wrapper(x_L,x_U,params_for_gp_toolbox.theta_vec, ...
                            params_for_gp_toolbox.sigma,z_i_L_vec,z_i_U_vec,bound_comp_opts,mu_sigma_bounds,'upper');

                        mu_sigma_bounds.sigma_u = sigma_star;
                        if ~isempty(x_sigma_star)
                            mu_sigma_bounds.x_sigma_u = x_sigma_star;
                        end
                        
                    else

                        [sigma_star,x_sigma_star,~,~] = compute_upper_lower_and_bound_sigma_sqe_wrapper(x_L,x_U,params_for_gp_toolbox.theta_vec, ...
                            params_for_gp_toolbox.sigma,z_i_L_vec,z_i_U_vec,bound_comp_opts,mu_sigma_bounds,'lower');
                        
                        mu_sigma_bounds.sigma_l = sigma_star;
                        if ~isempty(x_sigma_star)
                            mu_sigma_bounds.x_sigma_l = x_sigma_star;
                        end
                        
                    end
                    %3 - plug everything into erf function to get lower bound to the min.
                    out_l =  probit_likelihood_analytical_solution(mu_star,sigma_star,1);
                    %disp(mu_star)
                    %disp(sigma_star)
                    %disp(out_l)
                    %disp('-------')
                    x_candidates = [x_mu_star;x_sigma_star'];
                    
            
            end

        else
            error('Analytical Bound for likelihood function provided has not been implemented')
        end
    case 'discretised'
        %Case in which the integral is discretised
        [z_i_L_vec,z_i_U_vec] = pre_compute_z_intervals(x_L,x_U,params_for_gp_toolbox.theta_vec);
        %Computing minimum and maximum values for a-posteriori mean
        aux = tic;
        [mu_l,x_mu_l] = compute_lower_bound_mu_sqe(trainedSystem,x_L,x_U,params_for_gp_toolbox.theta_vec,...
            params_for_gp_toolbox.sigma,z_i_L_vec,z_i_U_vec);
        
        [mu_u,x_mu_u] = compute_upper_bound_mu_sqe(trainedSystem,x_L,x_U,params_for_gp_toolbox.theta_vec,...
            params_for_gp_toolbox.sigma,z_i_L_vec,z_i_U_vec,mu_l);
        mu_time = mu_time + toc(aux);
    
        %new implementation
        if bound_comp_opts.numberOfThreads > 1
            if isempty(gcp('nocreate'))
                parpool(bound_comp_opts.numberOfThreads);
            end
        end
        tic

        [sigma_l_new,x_sigma_l_new,sigma_u_new,x_sigma_u_new] = compute_upper_lower_and_bound_sigma_sqe_wrapper(x_L,x_U,params_for_gp_toolbox.theta_vec, ...
            params_for_gp_toolbox.sigma,z_i_L_vec,z_i_U_vec,bound_comp_opts,mu_sigma_bounds);
        if bound_comp_opts.iteration_count == 0 && strcmp(bound_comp_opts.var_bound,'slow')
            bound_comp_opts_quick = bound_comp_opts;
            bound_comp_opts_quick.var_bound = 'quick';
            [sigma_l_new_new,x_sigma_l_new_new,sigma_u_new_new,x_sigma_u_new_new] = compute_upper_lower_and_bound_sigma_sqe_wrapper(x_L,x_U,params_for_gp_toolbox.theta_vec, ...
            params_for_gp_toolbox.sigma,z_i_L_vec,z_i_U_vec,bound_comp_opts_quick,mu_sigma_bounds);
            if (sigma_u_new_new - sigma_l_new_new) <=  (sigma_u_new - sigma_l_new)
                bound_comp_opts = bound_comp_opts_quick;
                sigma_u_new = sigma_u_new_new;
                sigma_l_new = sigma_l_new_new;
                x_sigma_l_new = x_sigma_l_new_new;
                x_sigma_u_new = x_sigma_u_new_new;
            end
        end
        
        
        std_time = std_time + toc;
        sigma_star = pred_var(testIdx);
        
        if (bound_comp_opts.iteration_count == 0) && (sigma_l_new == 0)
        %if (sigma_l_new == 0)
            update_loop_vec(mu_l,mu_u,sigma_u_new);
        end
        %stop
        tic
        out_l = compute_pi_extreme(@normcdf,mu_l,mu_u,sigma_l_new,sigma_u_new,-4,4,bound_comp_opts.N,max_or_min);

        discrete_time = discrete_time + toc;
        %out_l = a;
        x_candidates = [x_mu_l;x_mu_u;];


        if ~isempty(x_sigma_l_new)  && all(x_sigma_l_new' >= x_L) && all(x_sigma_l_new' <= x_U)
            x_candidates = [x_candidates;x_sigma_l_new'];
        end
        if ~isempty(x_sigma_u_new) && all(x_sigma_u_new' >= x_L) && all(x_sigma_u_new' <= x_U)
            x_candidates = [x_candidates;x_sigma_u_new'];
        end
        if ~isempty(x_over) && all(x_over >= x_L) && all(x_over <= x_U)
            x_candidates = [x_candidates;x_over];
        end
        
        
        %mu_sigma_bounds.mu_l = mu_l;
        %mu_sigma_bounds.mu_u = mu_u;
        mu_sigma_bounds.sigma_l = sigma_l_new;
        if ~isempty(x_sigma_l_new)
            mu_sigma_bounds.x_sigma_l = x_sigma_l_new;
        end
        mu_sigma_bounds.sigma_u = sigma_u_new;
        if ~isempty(x_sigma_u_new)
            mu_sigma_bounds.x_sigma_u = x_sigma_u_new;
        end
end




%Plug promising input points into the GP model to get a valid upper
%bound to the min

tic
[~, ~, a, ~,lp] = gp(params_for_gp_toolbox.hyp, params_for_gp_toolbox.infFun, ...
                  params_for_gp_toolbox.meanfunc, params_for_gp_toolbox.covfunc, params_for_gp_toolbox.likfunc,...
                  training_data, post, x_candidates, ones(size(x_candidates,1), 1));
out = exp(lp);
%if ~mod(bound_comp_opts.iteration_count,100)
%    disp([mu_l,a(1);a(2),mu_u;sigma_l_new,nan;nan,sigma_u_new])
%end
inference_time = inference_time + toc;

if strcmp(max_or_min,'max')
    [~,idx] = max(out);
    out_u = out_l;
    out_l = out(idx);
else
    [out_u,idx] = min(out);
end
x_over_star = x_candidates(idx,:);

try
assert(out_l <= out_u)
catch
    disp('')
end
end

