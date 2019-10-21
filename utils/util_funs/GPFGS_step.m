function x_adv = GPFGS_step(xstar,epsilon,params_for_gp_toolbox,bound_comp_opts)
%Computes an adversarial example according to the GPFGS algorithm
%(Algorithm 1 in Appendix) of the paper 'The Limitations of Model
%Uncertainty in Adversarial Settings' by Grosse, K. et al. (2018)


%   xstar                   test point that is to be perturbed
%   epsilon                 magnitude of perturbation in L^inf norm
%   params_for_gp_toolbox   structure that contains GP parameters & specs



global post
global training_data

trainedSystem = post.alpha;
X_train = training_data;
theta = params_for_gp_toolbox.theta_vec;



gp_trained_params.kernel = 'sqe';
gp_trained_params.mle = false;
gp_trained_params.sigma = params_for_gp_toolbox.sigma;
gp_trained_params.theta_vec = params_for_gp_toolbox.theta_vec;

perturbation = zeros(size(xstar));

Kstar = getKernel(X_train,xstar,gp_trained_params); 
X_tilde = xstar-training_data;
mean_grad = 2*(-diag(theta)*X_tilde'*(Kstar.*trainedSystem))';  %factor 2 necessary as 
%our getKernel implementation lacks factor 0.5 -> otherwise does not match
%output of gp function

perturbation(bound_comp_opts.pix_2_mod) = epsilon*sign(mean_grad(bound_comp_opts.pix_2_mod));

if bound_comp_opts.constrain_2_one
    x_adv = min(max(0,xstar + perturbation),1);
else
    x_adv = xstar + perturbation;
end

end


%% Old code to check if derivative is correct

%global training_labels
%hyp = params_for_gp_toolbox.hyp; %%
%infFun = params_for_gp_toolbox.infFun; %%
%meanfunc = params_for_gp_toolbox.meanfunc; %%
%covfunc = params_for_gp_toolbox.covfunc; %%
%likfunc = params_for_gp_toolbox.likfunc; %%


%mean = ( Kstar'*trainedSystem);
%mean_grad1 = - mean * theta.* xstar;
%delta = 0.01
%num_grad = ones(1,size(training_data,2)); %%
%xdelta1 = xstar; %%
%xdelta2 = xstar; %%

%value_diffs = ones(11,2);

%for ii = 1:size(training_data,2)
%    xdelta1(ii) = xdelta1(ii) + delta; %%
%    xdelta2(ii) = xdelta2(ii) - delta; %%
%    Kdelta1 = getKernel(X_train,xdelta1,gp_trained_params); %%
%    Kdelta2 = getKernel(X_train,xdelta2,gp_trained_params); %%
%    mean_delta1 = ( Kdelta1'*trainedSystem); %%
%    mean_delta2 = ( Kdelta2'*trainedSystem); %%
    
    %disp(xdelta1)
        
%    [aa, bb, cc, dd, lplp, postpost] = gp(hyp, infFun, meanfunc, ...
%            covfunc, likfunc, training_data, training_labels, [xdelta1;xdelta2], [1,1]');
%    xdelta1(ii) = xdelta1(ii) - delta; %%
%    xdelta2(ii) = xdelta2(ii) + delta; %%
%    value_diffs(ii,1) = mean_delta1 - cc(1);
%    value_diffs(ii,2) = mean_delta2 - cc(2);
%    num_grad(ii) = (mean_delta1 - mean_delta2)/(2*delta); %%
    
    
%    mean_grad1(ii) = mean_grad1(ii) + theta(ii)*((training_data(:,ii).* Kstar)'* trainedSystem);
%end

%disp('mean value diff gp function vs. manual')
%disp(value_diffs)

%disp('num_grad')
%disp(num_grad)
%disp('mean_grad1')
%disp(mean_grad1)
%disp('mean_grad')
%disp(mean_grad)