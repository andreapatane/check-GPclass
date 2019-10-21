function [x_adv,perturbed_mean] = GPJM_step(xstar,epsilon,params_for_gp_toolbox,bound_comp_opts)
%Aims to compute an iteration step inside the while loop of the calculation of 
%an adversarial example according to the GPJM algorithm (Algorithm 2 in Appendix) 
%of the paper 'The Limitations of Model Uncertainty in Adversarial Settings' 
%by Grosse, K. et al. (2018). The oonly difference is that we limit the
%total perturbation to epsilon in the L infty norm.

%   xstar                   point that is to be perturbed by one step
%   epsilon                 value that is multiplied with biggest gradient
%                           to determine size of perturbation
%   params_for_gp_toolbox   structure that contains GP parameters & specs

global post
global training_data

trainedSystem = post.alpha;
X_train = training_data;
theta = params_for_gp_toolbox.theta_vec;

x_adv = xstar;

dim = max(size(x_adv));

gp_trained_params.kernel = 'sqe';
gp_trained_params.mle = false;
gp_trained_params.sigma = params_for_gp_toolbox.sigma;
gp_trained_params.theta_vec = params_for_gp_toolbox.theta_vec;

nn = 0;
perturbedlist = setdiff(1:dim,bound_comp_opts.pix_2_mod);

while nn < max(size(bound_comp_opts.pix_2_mod))
    Kstar = getKernel(X_train,x_adv,gp_trained_params); 
    X_tilde = x_adv-training_data;
    mean_grad = 2*(-diag(theta)*X_tilde'*(Kstar.*trainedSystem))';  %factor 2 necessary as 
%our getKernel implementation lacks factor 0.5 -> otherwise does not match
%output of gp function
    mean_grad(perturbedlist) = 0.0;
    [bb,ii] = max(abs(mean_grad));

    if bound_comp_opts.constrain_2_one
        if epsilon*sign(mean_grad(ii)) > 0.0
            x_adv(ii) = 1.0;
        elseif epsilon*sign(mean_grad(ii)) < 0.0
            x_adv(ii) = 0.0;
        end            
    else
        x_adv(ii) = x_adv(ii) + epsilon*sign(mean_grad(ii));
    end
    
    
    perturbedlist(nn+1) = ii;

    Kadv = getKernel(X_train,x_adv,gp_trained_params); 
    perturbed_mean =  ( Kadv'*trainedSystem);
    if sign(epsilon) == sign(perturbed_mean)
        break
    end 
    nn = nn + 1;
end

if norm((x_adv-xstar),Inf)  > abs(epsilon) + 0.000000001 %need to add this for numerical imprecision in comparison
    disp('GPJM: Error, adversarial example too far away from test point')

        
end