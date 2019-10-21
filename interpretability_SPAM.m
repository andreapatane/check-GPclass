%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Interpretability analysis on SPAM dataset%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

run('toolboxes/gpml-matlab-master/startup.m')
clear all %#ok<CLALL>
close all
clc
addpath(genpath('utils/'))


%Code parameters
numberOfThreads = 1;

%GP model parameters
num_trs = 1000;
num_tes = 200;
train_iter = 40;
likfunc = @likErf;
infFun = @infLaplace;

%Branch and bound parameters
ntpoints = 1; %increase to 50 for plot in the 
Testpoints = [1:ntpoints];
bound_comp_opts.constrain_2_one = false;
bound_comp_opts.max_iterations = 1000;
bound_comp_opts.tollerance = 0.001;
bound_comp_opts.N = 1000;
bound_comp_opts.numberOfThreads = numberOfThreads;
bound_comp_opts.var_lb_every_NN_iter = realmax;
bound_comp_opts.var_ub_every_NN_iter = realmax;
bound_comp_opts.var_ub_start_at_iter = realmax;
bound_comp_opts.var_lb_start_at_iter = realmax;
bound_comp_opts.min_region_size = 1e-10;
bound_comp_opts.var_bound = 'quick';
bound_comp_opts.mode = 'binarypi';
bound_comp_opts.likmode = 'analytical';
pix_2_mod = [1,2,3,4,5,6,7,8,9,10,11];
n_samples = 30;
epsilons = [0.02,1.0];


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



[X_train,y_train,X_test,y_test ] = load_spam_dataset(num_trs,num_tes);



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
acc = mymetric(y_test,exp(lp),'Acc');
disp(acc)
[trainedSystem,S,params_for_gp_toolbox] = build_gp_trained_params(hyp,num_trs,infFun,meanfunc,covfunc,likfunc);
disp('Done with training')
%%

if bound_comp_opts.numberOfThreads > 1
    if isempty(gcp('nocreate'))
        parpool(bound_comp_opts.numberOfThreads);
    end
end



%% Create output placeholder to be filled


classA = sum(a > 0)/num_tes;
classB = sum(a < 0)/num_tes;



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
clear X_train
clear y_train
clear Kstar Kstarstar data latent_variance_prediction

S = S*params_for_gp_toolbox.sigma;
global R_inv
R_inv = S;
global U
global Lambda
R_inv = 0.5*(R_inv + R_inv');
[U,Lambda] = eig(R_inv);
Lambda = diag(Lambda);

for dd = 1:length(Testpoints)
    
    testIdx = Testpoints(dd);
    testPoint = X_test(testIdx,:);
    
    pis(1,dd) = exp(lp(testIdx));
    
    bound_comp_opts.epsilon = epsilons(1);
    disp(['epislon = ', num2str(bound_comp_opts.epsilon) ]);
    aux_time = tic;
    for ii = 1:length(pix_2_mod)
        disp(['Interpretability for dimension: ', int2str(ii)])
        [pi_LL,pi_UU,pi_LLm,pi_UUm] = interpretability_main(pix_2_mod(ii),bound_comp_opts,testPoint,testIdx,params_for_gp_toolbox,trainedSystem);
        pi_LLs_a(ii,dd) = pi_LL;
        pi_UUs_a(ii,dd) = pi_UU;
        pi_LLs_m(ii,dd) = pi_LLm;
        pi_UUs_m(ii,dd) = pi_UUm;
        
    end
    bound_time = bound_time + toc(aux_time);
    
    
    bound_comp_opts.epsilon = epsilons(2);
    disp(['epislon = ', num2str(bound_comp_opts.epsilon) ]);
    aux_time = tic;
    for ii = 1:length(pix_2_mod)
        disp(['Interpretability for dimension: ', int2str(ii)])
        [pi_LL,pi_UU,pi_LLm,pi_UUm] = interpretability_main(pix_2_mod(ii),bound_comp_opts,testPoint,testIdx,params_for_gp_toolbox,trainedSystem);
        pi_LLs_a2(ii,dd) = pi_LL;
        pi_UUs_a2(ii,dd) = pi_UU;
        pi_LLs_m2(ii,dd) = pi_LLm;
        pi_UUs_m2(ii,dd) = pi_UUm;
        
    end
    bound_time2 = bound_time2 + toc(aux_time);
    
    
    % LIME
    aux_time = tic;
    limeout = LimeForGP(testPoint,n_samples*11,'reg',params_for_gp_toolbox);
    limecoefs = limecoefs + limeout.coefs;
    limeMSE =  limeMSE + limeout.MSE;
    lime_time = lime_time + toc(aux_time);
    
    disp(strcat('DONE WITH TESTPOINT ', num2str(testIdx)))
    
end

disp(bound_time)

%% Display summary of results

diff_vec_a = (0.5/classA*round(pis,0) +  0.5/classB*(1-round(pis,0))).*(pi_UUs_a - pi_UUs_m + (pi_LLs_a-pi_LLs_m));

diff_vec_s = ((pi_Us_s - pis) - (pis - pi_Ls_s));

diff_vec_a2 = (0.5/classA*round(pis,0) +  0.5/classB*(1-round(pis,0))).*(pi_UUs_a2 - pi_UUs_m2 + (pi_LLs_a2-pi_LLs_m2));

diff_vec_s2 = ((pi_Us_s2 - pis) - (pis - pi_Ls_s2));

disp('mean of metric for analytical bounds small epsilon')
disp(mean(diff_vec_a,2))

disp('mean of metric for analytical bounds big epsilon')
disp(mean(diff_vec_a2,2))

disp('lime metric')
disp(limecoefs)

%%

max_abs_a = max(abs(mean(diff_vec_a,2)));
max_abs_a2 = max(abs(mean(diff_vec_a2,2)));
max_lime = max(abs(limecoefs));


