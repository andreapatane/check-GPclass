function [sigma_l,x_sigma_l,sigma_u,x_sigma_u] = compute_upper_lower_and_bound_sigma_sqe_quad_prog(x_L,x_U,theta_vec,sigma_prior,z_i_L_vec,z_i_U_vec,bound_comp_opts,mu_sigma_bounds,low_or_up)
if nargin < 9
    low_or_up = 'both';
end
global training_data
global R_inv
global U
global Lambda

r_L = exp(-z_i_U_vec);
r_U = exp(-z_i_L_vec);

fun_ln = @(r)(log(r));
d_fun_ln = @(r)(1./r);

n = length(r_L);

a_vec = zeros(2*n,1);
m = length(x_L);
B = zeros(2*n,n + m);
b_i_vec_L = zeros(1,m);
b_i_vec_U = zeros(1,m);

flagUpdateUB = ~mod(bound_comp_opts.iteration_count,bound_comp_opts.var_ub_every_NN_iter);
flagUpdateLB = ~mod(bound_comp_opts.iteration_count,bound_comp_opts.var_lb_every_NN_iter);
if bound_comp_opts.iteration_count > 0
    flagUpdateUB = flagUpdateUB && bound_comp_opts.iteration_count >= bound_comp_opts.var_ub_start_at_iter;
    flagUpdateLB = flagUpdateLB && bound_comp_opts.iteration_count >= bound_comp_opts.var_lb_start_at_iter;
end

flagUpdateUB = flagUpdateUB || ~isfield(mu_sigma_bounds,'sigma_u');
flagUpdateLB = flagUpdateLB || ~isfield(mu_sigma_bounds,'sigma_l');

switch low_or_up
    case 'both'
        true;
    case 'upper'
        flagUpdateLB = false;
    case 'lower'
        flagUpdateUB = false;
end

flagUpdateBoth = flagUpdateUB && flagUpdateLB;
%flagParallel = bound_comp_opts.numberOfThreads > 1;
flagParallel = false; %not worth it...
temp1 = 0;
temp2 = 0;
temp3 = 0;

if flagUpdateUB || flagUpdateLB
    %tic
    for ii = 1:n
        [a_i_L,b_i_L,a_i_U,b_i_U] = concave_bounds(r_L(ii),r_U(ii), fun_ln, d_fun_ln);
        %if b_i_L > 1.e28 || b_i_U > 1.e28
        %    disp('')
        %end
        a_ij_L_sum = 0;
        a_ij_U_sum = 0;
        for jj = 1:m
            if x_L(jj) < x_U(jj)
                fun_quad_j = @(x)( theta_vec(jj) * (x - training_data(ii,jj) )^2   );
                d_fun_quad_j = @(x)( 2*theta_vec(jj) * (x - training_data(ii,jj) )   );
                [a_ij_L,b_ij_L,a_ij_U,b_ij_U] = convex_bounds(x_L(jj),x_U(jj), fun_quad_j, d_fun_quad_j);
            else
                a_ij_L = theta_vec(jj) * (x_L(jj) - training_data(ii,jj) )^2;
                a_ij_U = a_ij_L;
                b_ij_L = 0;
                b_ij_U = 0;
            end
            
            b_i_vec_L(jj) = b_ij_L;
            b_i_vec_U(jj) = b_ij_U;
            a_ij_L_sum = a_ij_L_sum + a_ij_L;
            a_ij_U_sum = a_ij_U_sum + a_ij_U;
            
        end
        a_vec((2*ii)-1) = a_i_L + a_ij_L_sum;
        a_vec(2*ii) = - a_i_U - a_ij_U_sum;
        B((2*ii)-1,1:m) = b_i_vec_L;
        B(2*ii,1:m) = -b_i_vec_U;
        B((2*ii)-1,m+ii) = b_i_L;
        B(2*ii,m+ii) = - b_i_U;
    end
    %toc
end

if flagUpdateUB
    %disp('Recomputing variance upper bound...')
    Q = [zeros(m,m), zeros(m,n);
        zeros(n,m),R_inv];
    
    opts = optimoptions('quadprog');
    opts.Display = 'off';
    opts.MaxIterations = 100;
    opts.ConstraintTolerance = 1.0e-16;
    opts.StepTolerance = 1.0e-03;
    opts.OptimalityTolerance = 1.0e-03;
    %[x_opt,lb,exitFlag] = quadprog(2*Q,zeros(n+m,1),B,-a_vec,[],[],[x_L,r_L],[x_U,r_U],[],opts);
    %If I get rid off of the the linear constraints here it seems to obtain
    %similar results, but 100 times faster. Consider using the line above if we
    %see that this doesn't work.
    %[x_opt,ub,exitFlag] = quadprog(2*Q,zeros(n+m,1),[],[],[],[],[x_L,r_L],[x_U,r_U],[],opts);
    if ~flagUpdateBoth || ~flagParallel
        aux = tic;
        constTols = [1.0e-16, 1.0e-10, 1.0e-6, 1.0e-3];
        for ii = 1:length(constTols)
            
            opts.ConstraintTolerance = constTols(ii);
            [x_opt,ub,exitFlag] = quadprog(sparse(2*Q),[],B,-a_vec,[],[],[x_L,r_L],[x_U,r_U],[],opts);
            if exitFlag ~= -2
                break;
            end
        end
        temp1 = toc(aux);
        
        try
            
            switch exitFlag
                case -2
                    error('problem unfeasible?! This should not happen really...')
                case -3
                    error('Unbounded problem?! This should not happen really...')
                case 2
                    disp('small step, but I was still slowly improving...')
                case -6
                    error('Non-convex problem... Check for numerical cancellation errors')
                case -8
                    error('Some random stuff related to step direction. Please do not occur!')
            end
            
            %FIX this in case of point not found!
            sigma_u = sigma_prior*(1 - ub);
            x_sigma_u = x_opt(1:m);
            x_sigma_u = max(x_sigma_u,x_L');
            x_sigma_u = min(x_sigma_u,x_U');
            if isfield(mu_sigma_bounds,'sigma_u')
                sigma_u = min(sigma_u,mu_sigma_bounds.sigma_u);
            end
            %
            
        catch
            
            if isfield(mu_sigma_bounds,'sigma_u')
                sigma_u = mu_sigma_bounds.sigma_u;
            else
                %sigma_u = [];
                error(['sigma_u not yet cached. The optimisation failed at the first branch and bound iteration',...
                    'consider reducing the constraints tolerance for this specific test point'])
            end
            if isfield(mu_sigma_bounds,'x_sigma_u')
                x_sigma_u = mu_sigma_bounds.x_sigma_u;
            else
                x_sigma_u = [];
            end
            
        end
    end
    
else
    if isfield(mu_sigma_bounds,'sigma_u')
        sigma_u = mu_sigma_bounds.sigma_u;
    else
        sigma_u = [];
    end
    if isfield(mu_sigma_bounds,'x_sigma_u')
        x_sigma_u = mu_sigma_bounds.x_sigma_u;
    else
        x_sigma_u = [];
    end
end



if flagUpdateLB
    
    %disp('Recomputing variance lower bound...')
    %tic
    %[r_L_hat_old,r_U_hat_old] = get_r_hat_limits_old(r_L,r_U,B,a_vec,x_L,x_U);
    %toc
    aux = tic;
    [r_L_hat,r_U_hat] = get_r_hat_limits(r_L,r_U,B,a_vec,x_L,x_U);
    temp2 = toc(aux);
    
    B_lin = B;
    B_lin(:,(m+1):end) = B_lin(:,(m+1):end)*U;
    
    b_lin_obj_vec = zeros(n,1);
    a_lin_obj_sum = 0;
    
    for ii = 1:length(b_lin_obj_vec)
        fun_diag_quad_i = @(r)(- Lambda(ii)*r.^2);
        d_fun_diag_quad_i = @(r)(- 2*Lambda(ii)*r);
        [a_i_L,b_lin_obj_vec(ii),~,~] = concave_bounds(r_L_hat(ii),r_U_hat(ii), fun_diag_quad_i, d_fun_diag_quad_i);
        a_lin_obj_sum = a_lin_obj_sum + a_i_L;

    end
    
    
    opts_lin = optimoptions('linprog');
    opts_lin.Display = 'off';
    opts_lin.Preprocess = 'basic';
    opts_lin.Algorithm = 'interior-point';
    opts_lin.OptimalityTolerance = 1e-3;

    aux = tic;
    a_vec_hat = - a_vec - B_lin(:,1:m)*x_L' - B_lin(:,(m+1):end)*r_L_hat';
    B_lin(:,1:m) = B_lin(:,1:m).*((x_U - x_L));
    B_lin(:,(m+1):end) = B_lin(:,(m+1):end).*(r_U_hat - r_L_hat);
    

    
    constTols = [1.0e-6, 1.0e-4 , 1.0e-3 , 1.0e-2, 1.0e-1 ];
    for ii = 1:length(constTols)
        opts_lin.ConstraintTolerance = constTols(ii);
        opts_lin.Algorithm = 'interior-point';
        [x_opt,lb_hat,exitFlag] = linprog([zeros(m,1);b_lin_obj_vec.*(r_U_hat - r_L_hat)'],B_lin,a_vec_hat,[],[],zeros(m+n,1),ones(m+n,1),opts_lin);
        if exitFlag < 0
            opts_lin.Algorithm = 'interior-point-legacy';
            [x_opt,lb_hat,exitFlag] = linprog([zeros(m,1);b_lin_obj_vec.*(r_U_hat - r_L_hat)'],B_lin,a_vec_hat,[],[],zeros(m+n,1),ones(m+n,1),opts_lin);
        end
        if exitFlag < 0
            opts_lin.Algorithm = 'interior-point';
            [x_opt,lb_hat,exitFlag] = linprog([zeros(m,1);b_lin_obj_vec.*(r_U_hat - r_L_hat)'],B_lin(all(B_lin < 1.0e10,2),:),a_vec_hat(all(B_lin < 1.0e10,2)),[],[],zeros(m+n,1),ones(m+n,1),opts_lin);
        end
        if exitFlag >= 0
            break;
        end
    end
    
    
   
    try
        switch exitFlag
            case 0
                disp('number of iterations for quadprog exceeded')
            case -2
                error('problem unfeasible?! This should not happen really...')
            case -3
                error('Unbounded problem?! This should not happen really...')
            case -4
                disp('weird NaN obtained...')
            case -4
                disp('Unfeasible and/or unbounded?')
            case -6
                error('The problem is ill-posed or badly conditioned.')
        end
        lb = lb_hat + a_lin_obj_sum +  r_L_hat*b_lin_obj_vec;
        temp3 = toc(aux);
        sigma_l = sigma_prior*(1 + lb);
        x_sigma_l = x_opt(1:m);
        x_sigma_l = max(x_sigma_l,x_L');
        x_sigma_l = min(x_sigma_l,x_U');
        sigma_l = max(sigma_l,0);
        if isfield(mu_sigma_bounds,'sigma_l')
            sigma_l = max(sigma_l,mu_sigma_bounds.sigma_l);
        end
    catch
        
        if isfield(mu_sigma_bounds,'sigma_l')
            sigma_l = mu_sigma_bounds.sigma_l;
        else
            error(['sigma_l not yet cached. The optimisation failed at the first branch and bound iteration',...
                'consider reducing the constraints tolerance for this specific test point'])
        end
        if isfield(mu_sigma_bounds,'x_sigma_l')
            x_sigma_l = mu_sigma_bounds.x_sigma_l;
        else
            x_sigma_l = [];
        end
    end

    
    
else
    if isfield(mu_sigma_bounds,'sigma_l')
        sigma_l = mu_sigma_bounds.sigma_l;
    else
        sigma_l = [];
    end
    if isfield(mu_sigma_bounds,'x_sigma_l')
        x_sigma_l = mu_sigma_bounds.x_sigma_l;
    else
        x_sigma_l = [];
    end
end
end
