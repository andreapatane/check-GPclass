%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Interpretability analysis on Synthetic2D dataset%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

run('toolboxes/gpml-matlab-master/startup.m')
clear all %#ok<CLALL>
close all
clc
addpath(genpath('utils/'))


%Code parameters
numberOfThreads = 1;

%GP model parameters
likfunc = @likErf; %link function
infFun = @infLaplace; %inference method
num_trs = 1000; %number of training points
num_tes = 200; %number of test points
train_iter = 40;


%Branch and bound parameters
epsilons = [0.02,2.0];
bound_comp_opts.constrain_2_one = false;
bound_comp_opts.max_iterations = 1000;
bound_comp_opts.tollerance = 0.001;
bound_comp_opts.N = 1000;
bound_comp_opts.numberOfThreads = numberOfThreads;
bound_comp_opts.var_lb_every_NN_iter = realmax;
bound_comp_opts.var_ub_every_NN_iter = realmax;
bound_comp_opts.var_ub_start_at_iter = realmax;
bound_comp_opts.var_lb_start_at_iter = realmax;
bound_comp_opts.min_region_size = 1e-20;
bound_comp_opts.var_bound = 'quick';
bound_comp_opts.mode = 'binarypi';
bound_comp_opts.likmode = 'analytical';
Testpoints = [81];
pix_2_mod = [1,2];
n_samples = 30;



%some global variables used for time monitoring of the method
global mu_time
global std_time
global discrete_time
global inference_time
discrete_time = 0;
mu_time = 0;
std_time = 0;
inference_time = 0;
sample_time = 0;
bound_time = 0;
bound_time2 = 0;
lime_time = 0;


global pred_var
global post


rng(1)
maxNumCompThreads(numberOfThreads);


%Load data
[X_train,y_train,X_test,y_test] = generate_2d_synthetic_datasets(num_trs,num_tes);


%% training of the GP
disp('Training GP')
meanfunc = @meanZero;
covfunc = @covSEard;
ell = 1.0;
sf = 1.0;
hyp.cov = log([ones(1,size(X_train,2))*ell, sf]);
hyp = minimize(hyp, @gp, -train_iter, infFun, meanfunc, covfunc, likfunc, X_train, y_train);
[a, b, c, pred_var, lp, post] = gp(hyp, infFun, meanfunc, ...
    covfunc, likfunc, X_train, y_train, X_test, ones(size(X_test,1), 1));

[trainedSystem,S,params_for_gp_toolbox] = build_gp_trained_params(hyp,num_trs,infFun,meanfunc,covfunc,likfunc);

disp('Done with training')
%%

if bound_comp_opts.numberOfThreads > 1
    if isempty(gcp('nocreate'))
        parpool(bound_comp_opts.numberOfThreads);
    end
end




classA = sum(a > 0)/num_tes;
classB = sum(a < 0)/num_tes;


%% Create output placeholder to be filled
pis = zeros(1,length(Testpoints));
pi_LLs_a = zeros(length(pix_2_mod),length(Testpoints));
pi_UUs_a = zeros(length(pix_2_mod),length(Testpoints));
pi_LLs_m = zeros(length(pix_2_mod),length(Testpoints));
pi_UUs_m = zeros(length(pix_2_mod),length(Testpoints));
pi_Us_s = zeros(length(pix_2_mod),length(Testpoints));
pi_Ls_s = zeros(length(pix_2_mod),length(Testpoints));
pi_LLs_a2 = zeros(length(pix_2_mod),length(Testpoints));
pi_UUs_a2 = zeros(length(pix_2_mod),length(Testpoints));
pi_LLs_m2 = zeros(length(pix_2_mod),length(Testpoints));
pi_UUs_m2 = zeros(length(pix_2_mod),length(Testpoints));
pi_Us_s2 = zeros(length(pix_2_mod),length(Testpoints));
pi_Ls_s2 = zeros(length(pix_2_mod),length(Testpoints));



limecoefs = zeros(size(X_train,2),1);
limeMSE = 0.0;

%% For loop over test points
global training_data
global training_labels
global loop_vec2
training_data = X_train;
training_labels = y_train;
loop_vec2 = discretise_real_line(bound_comp_opts.N);
clear Kstar Kstarstar data latent_variance_prediction X_train y_train

S = S*params_for_gp_toolbox.sigma;
global R_inv
global U
global Lambda
R_inv = S;
R_inv = 0.5*(R_inv + R_inv');
[U,Lambda] = eig(R_inv);
Lambda = diag(Lambda);

for dd = 1:length(Testpoints)
    
    testIdx = Testpoints(dd);
    testPoint = X_test(testIdx,:);
    pis(1,dd) = exp(lp(testIdx));
 
    bound_comp_opts.epsilon = epsilons(1);
    
    aux_time = tic;
    for ii = 1:length(pix_2_mod)
        [pi_LL,pi_UU,pi_LLm,pi_UUm] = interpretability_main(pix_2_mod(ii),bound_comp_opts,testPoint,testIdx,params_for_gp_toolbox,trainedSystem);
        pi_LLs_a(ii,dd) = pi_LL;
        pi_UUs_a(ii,dd) = pi_UU;
        pi_LLs_m(ii,dd) = pi_LLm;
        pi_UUs_m(ii,dd) = pi_UUm;
    end
    bound_time = bound_time + toc(aux_time);
    
    bound_comp_opts.epsilon = epsilons(2);
    
    aux_time = tic;
    for ii = 1:length(pix_2_mod)
        [pi_LL,pi_UU,pi_LLm,pi_UUm] = interpretability_main(pix_2_mod(ii),bound_comp_opts,testPoint,testIdx,params_for_gp_toolbox,trainedSystem);
        pi_LLs_a2(ii,dd) = pi_LL;
        pi_UUs_a2(ii,dd) = pi_UU;
        pi_LLs_m2(ii,dd) = pi_LLm;
        pi_UUs_m2(ii,dd) = pi_UUm;
    end
    
    bound_time2 = bound_time2 + toc(aux_time);
    
    
    % LIME
    aux_time = tic;
    limeout = LimeForGP(testPoint,n_samples*2,'reg',params_for_gp_toolbox);
    limecoefs = limecoefs + limeout.coefs;
    limeMSE =  limeMSE + limeout.MSE;
    lime_time = lime_time + toc(aux_time);
    disp(strcat('DONE WITH TESTPOINT ', num2str(testIdx)))
    
end

disp(bound_time)



