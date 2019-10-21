function [pi_LL,pi_UU,pi_LU,pi_UL,iter_count,exitFlag] = main_pi_hat_computation(mode,testPoint,testIdx,...
params_for_gp_toolbox,bound_comp_opts,varargin)

% --exitFlags coming from branch and bound --
% 0 - did not run bnb
% 1 - terminating condition met by comparing lb_b4_star to ubstar
% 11 - terminating condition met while picking next region
% 12 - terminating condition met because no more regions to check
% 2 - region minimum size undercut
% 3 - max iterations reached
% 4 - branch and bound stagnated



global theta_vec_train_squared
global training_data

if mode == 'all'
    minmode = true;
    maxmode = true;
elseif mode == 'min'
    minmode = true;
    maxmode = false;
elseif mode == 'max'
    minmode = false;
    maxmode = true;
else
    disp('Mode specified incorrectly')
    return
end


%in the multi-class case, if you wanna have different kernels you need to
%put an additional for loop around this block of code
theta_vec_train_squared = zeros(size(training_data,1),1);
for ii = 1:size(training_data,1)
    theta_vec_train_squared(ii) = dot(params_for_gp_toolbox.theta_vec,training_data(ii,:).^2);
end




if isfield(bound_comp_opts,'x_L')
    x_L = bound_comp_opts.x_L;
    x_U = bound_comp_opts.x_U;                                        
else
    [x_L, x_U] = compute_hyper_rectangle(bound_comp_opts.epsilon,testPoint,...
        bound_comp_opts.pix_2_mod,bound_comp_opts.constrain_2_one);    
end



%%%% Here I perform bnb for the minimum of pi_hat
if minmode
    disp('Min computation:')
    [pi_LL, pi_LU, count, flag] = bnb_for_gp_classification(testIdx,x_L,x_U,params_for_gp_toolbox,'min',bound_comp_opts.mode,bound_comp_opts,varargin{:});
iter_count.min = count;
exitFlag.min = flag;
else
    iter_count.min = 0;
    pi_LL = 0.0;
    pi_LU = 0.5;
    exitFlag.min = 0;
end

%%% Then I perform bnb for the maximasation of pi_hat
if maxmode
    disp('Max computation:')
    [pi_UL, pi_UU, count, flag ] = bnb_for_gp_classification(testIdx,x_L,x_U,params_for_gp_toolbox,'max',bound_comp_opts.mode,bound_comp_opts,varargin{:});
    iter_count.max = count;
    exitFlag.max = flag;
else
    iter_count.max = 0;
    pi_UL = 0.5;
    pi_UU = 1.0;
    exitFlag.max = 0;
end



end
