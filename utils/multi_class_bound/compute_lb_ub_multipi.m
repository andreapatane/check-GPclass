function [out_l,out_u,x_over_star,mu_sigma_bounds,bound_comp_opts] = compute_lb_ub_multipi(testIdx,max_or_min,...
    kernel_name,x_L,x_U,params_for_gp_toolbox,bound_comp_opts,trainedSystem,mu_sigma_bounds,x_over)

global multi_grid_extreme multi_grid
global training_data R_inv
global std_time mu_time discrete_time inference_time

assert(strcmp(kernel_name,'sqe'))
switch bound_comp_opts.likmode
    case 'discretised'
        %upper and lower bound on kernel auxilary function
        x_candidates = [];
        [z_i_L_vec,z_i_U_vec] = pre_compute_z_intervals(x_L,x_U,params_for_gp_toolbox.theta_vec); %if you have different hyer-parameters for different classes you need to put a for loop around this line
        
        %Computing upper and lower bound on mean
        mu_l = zeros(1,params_for_gp_toolbox.nout);
        mu_u = zeros(1,params_for_gp_toolbox.nout);
        %
        aux = tic;
        for ii = 1:params_for_gp_toolbox.nout
            [mu_l(ii),x_mu_l] = compute_lower_bound_mu_sqe(trainedSystem(:,ii),x_L,x_U,params_for_gp_toolbox.theta_vec,...
                params_for_gp_toolbox.sigma,z_i_L_vec,z_i_U_vec);
            %
            [mu_u(ii),x_mu_u] = compute_upper_bound_mu_sqe(trainedSystem(:,ii),x_L,x_U,params_for_gp_toolbox.theta_vec,...
                params_for_gp_toolbox.sigma,z_i_L_vec,z_i_U_vec,mu_l(ii));
            x_candidates = [x_candidates;x_mu_l;x_mu_u];
        end
        mu_time = mu_time + toc(aux);
        n_tr = size(trainedSystem,1);
        
        sigma_l_new = zeros(params_for_gp_toolbox.nout,params_for_gp_toolbox.nout);
        sigma_u_new = zeros(params_for_gp_toolbox.nout,params_for_gp_toolbox.nout);
        
        aux = tic;
        for ii = 1:params_for_gp_toolbox.nout
            for jj = ii:params_for_gp_toolbox.nout
                offset_rows = (ii-1)*n_tr;
                offset_cols = (jj-1)*n_tr;
                %if you wanna have different hyperparameters for different
                %classes you need to modify the function below in order to
                %take to two vectors for the kernel hyperparameters
                
                [sigma_l_new(ii,jj),x_sigma_l_new,sigma_u_new(ii,jj),x_sigma_u_new] = compute_upper_lower_and_bound_sigma_sqe_wrapper(x_L,x_U,params_for_gp_toolbox.theta_vec, ...
                    params_for_gp_toolbox.sigma,z_i_L_vec,z_i_U_vec,bound_comp_opts,mu_sigma_bounds,'both',offset_rows,offset_cols);
                x_candidates = [x_candidates;x_sigma_l_new';x_sigma_u_new'];
            end
        end
        sigma_l_new = sigma_l_new + sigma_l_new' - diag(diag(sigma_l_new));
        sigma_u_new = sigma_u_new + sigma_u_new' - diag(diag(sigma_u_new));
        std_time = std_time + toc(aux);
        if bound_comp_opts.iteration_count == 0
            lowers = mu_l - 1*diag(sigma_l_new)';
            uppers = mu_u + 1*diag(sigma_u_new)';
            multi_grid = discretise_real_vec_space(bound_comp_opts.N,lowers,uppers,params_for_gp_toolbox.nout);
            if strcmp(max_or_min,'max')
                [~,multi_grid_extreme] = find_softmax_extrema_in_grid(bound_comp_opts.classIdx);
            elseif strcmp(max_or_min,'min')
                [multi_grid_extreme,~] = find_softmax_extrema_in_grid(bound_comp_opts.classIdx);
            end
        end
        aux = tic;
        out_l = bound_pi_multiclass(mu_l,mu_u,sigma_l_new,sigma_u_new,max_or_min);
        discrete_time = discrete_time + toc(aux);
        if ~isempty(x_over) && all(x_over >= x_L) && all(x_over <= x_U)
            x_candidates = [x_candidates;x_over];
        end
        aux = tic;
        [out,~,~,~] = gp_multiclass_manual_prediction(x_candidates,training_data,params_for_gp_toolbox,trainedSystem,R_inv/params_for_gp_toolbox.sigma);
        inference_time = inference_time + toc(aux);
        out = exp(out(:,bound_comp_opts.classIdx));
        
        if strcmp(max_or_min,'max')
            [~,idxOpt] = max(out);
            out_u = out_l;
            out_l = out(idxOpt);
        else
            [out_u,idxOpt] = min(out);
        end
        x_over_star = x_candidates(idxOpt,:);
    otherwise
        error('Likmode not implemented for the multiclass case')
        
end


end