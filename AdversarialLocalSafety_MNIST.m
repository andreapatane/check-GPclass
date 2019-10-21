%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Adversarial safety for MNIST (Figure 3 Left Column of paper)%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%(This will take a bit to run...)%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

run('toolboxes/gpml-matlab-master/startup.m')
clear all %#ok<CLALL>
close all
clc

addpath(genpath('utils/'))


numberOfThreads = 2 ;

%global GPC model parameters
num_trs = 1000;
num_tes = 200;
likfunc = @likErf;
infFun = @infLaplace;
downsamplingFlag = true;
train_iter = 10;

%Test points selected
Testpoints = [24];

%Branch and bound parameters
epsilons = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0];
n_eps = length(epsilons);
step_size = 0.001;
n_steps = 1;
bound_comp_opts.mod_modus = 'sift';
bound_comp_opts.pix_2_mod = [];
bound_comp_opts.constrain_2_one = true;
bound_comp_opts.max_iterations = 30000;
bound_comp_opts.tollerance = 0.02;
bound_comp_opts.N = 1000;
bound_comp_opts.numberOfThreads = numberOfThreads;
bound_comp_opts.var_lb_every_NN_iter = realmax;
bound_comp_opts.var_ub_every_NN_iter = realmax;
bound_comp_opts.var_ub_start_at_iter = realmax;
bound_comp_opts.var_lb_start_at_iter = realmax;
bound_comp_opts.min_region_size = 1e-20;
bound_comp_opts.var_bound = 'slow';
bound_comp_opts.likmode = 'analytical';
bound_comp_opts.mode = 'binarypi';
n_samples = 1000;

rng(1)
maxNumCompThreads(numberOfThreads);


global mu_time
global std_time
global discrete_time
global inference_time
global pred_var

global post
discrete_time = 0;
mu_time = 0;
std_time = 0;
inference_time = 0;

%% GPC training
[X_train,y_train,X_test,y_test ] = load_mnist38(num_trs,num_tes,downsamplingFlag);

%plotting selected test point
imshow(reshape(X_test(Testpoints,:),14,14),'InitialMagnification' ,'fit')
%

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

%Praparing results structure and matrixes
defense.analytical = zeros(1+length(Testpoints),n_eps);
defense.analytical(2:end,1) = exp(lp(Testpoints));
defense.analytical(1,:) = epsilons;
defense.analytical_exact = zeros(1+length(Testpoints),n_eps);
defense.analytical_exact(2:end,1) = exp(lp(Testpoints));
defense.analytical_exact(1,:) = epsilons;
iter_count.analytical = zeros(1+length(Testpoints),n_eps);
iter_count.analytical(1,:) = epsilons;
flags.analytical = zeros(1+length(Testpoints),n_eps);
flags.analytical(1,:) = epsilons;

attacks_GPFGS.probs = zeros(1+length(Testpoints),n_eps);
attacks_GPFGS.probs(2:end,1) = exp(lp(Testpoints));
attacks_GPFGS.probs(1,:) = epsilons;

modified_pixel = zeros(length(Testpoints),20);
min_breakable_vs_broken = [ones(length(Testpoints),1),ones(length(Testpoints),1),ones(length(Testpoints),1),ones(length(Testpoints),1)];


%Adversarial Robustness analysis of selected test point

global training_data
global training_labels
global loop_vec2
training_data = X_train;
training_labels = y_train;
loop_vec2 = discretise_real_line(bound_comp_opts.N);
clear X_train
clear Kstar Kstarstar data latent_variance_prediction

S = S*params_for_gp_toolbox.sigma;
global R_inv
global U
global Lambda
R_inv = S;
R_inv = 0.5*(R_inv + R_inv');
[U,Lambda] = eig(R_inv);
Lambda = diag(Lambda);

ccount = 1;
dd = 1;
bound_comp_opts.iteration_count = 0;
testIdx = Testpoints(dd);
testPoint = X_test(testIdx,:);


pi_LLs_a = [exp(lp(testIdx)),-1:-1:-(n_eps-1)];
pi_UUs_a = [exp(lp(testIdx)),2:n_eps];
pi_LUs_a = [exp(lp(testIdx)),-1:-1:-(n_eps-1)];
pi_ULs_a = [exp(lp(testIdx)),2:n_eps];

% Compute relevant pixels
if strcmp(bound_comp_opts.mod_modus,'sift')
    pix_2_mod = sift_on_mnist(testPoint,1,10);
    bound_comp_opts.pix_2_mod = pix_2_mod{1};
elseif strcmp(bound_comp_opts.mod_modus,'ls')
    [~,bound_comp_opts.pix_2_mod] = maxk(params_for_gp_toolbox.theta_vec,3);
else
    true;
end

modified_pixel(dd,1:length(bound_comp_opts.pix_2_mod)) = bound_comp_opts.pix_2_mod;

%  For loop over epsilons to get pi_Ls and pi_Us
for ii = 2:n_eps
    bound_comp_opts.epsilon = epsilons(ii);
    disp('Current epsilon')
    disp(bound_comp_opts.epsilon)
    
    
    %%-------- GPFGS Attacks --------
    
    [attacks_GPFGS.probs(dd+1,ii),class1,maxminmode] = GPFGS_attack(lp(testIdx),epsilons(ii),step_size,testPoint,n_steps,params_for_gp_toolbox,bound_comp_opts);
    
    
    
    if class1 && (attacks_GPFGS.probs(dd+1,ii) < 0.5) && (min_breakable_vs_broken(dd,2) == 1)
        min_breakable_vs_broken(dd,2) = epsilons(ii);
    elseif ~class1 && ( attacks_GPFGS.probs(dd+1,ii) >= 0.5) && (min_breakable_vs_broken(dd,2) == 1)
        min_breakable_vs_broken(dd,2) = epsilons(ii);
    end
    
    %%-------- Verification --------
    
    [x_L, x_U] = compute_hyper_rectangle(bound_comp_opts.epsilon,testPoint,...
        bound_comp_opts.pix_2_mod,bound_comp_opts.constrain_2_one);
    bound_comp_opts.x_L = x_L;
    bound_comp_opts.x_U = x_U;
    
    
    iter_count.analytical(dd+1,1) = testIdx;
    flags.analytical(dd+1,1) = testIdx;
    aux = tic;
    [pi_LL,pi_UU, pi_LU,pi_UL,count,exitFlag] = main_pi_hat_computation(maxminmode,testPoint,testIdx,...
        params_for_gp_toolbox,bound_comp_opts,trainedSystem,S);
    pi_LLs_a(ii) = pi_LL;
    pi_UUs_a(ii) = pi_UU;
    pi_LUs_a(ii) = pi_LU;
    pi_ULs_a(ii) = pi_UL;
    toc(aux)
    if class1
        defense.analytical(dd+1,ii) = pi_LL;
        defense.analytical_exact(dd+1,ii) = pi_LU;
        iter_count.analytical(dd+1,ii) = count.min;
        flags.analytical(dd+1,ii) = exitFlag.min;
    else
        defense.analytical(dd+1,ii) = pi_UU;
        defense.analytical_exact(dd+1,ii) = pi_UL;
        iter_count.analytical(dd+1,ii) = count.max;
        flags.analytical(dd+1,ii) = exitFlag.max;
    end
    
    
    if class1 && (pi_LLs_a(ii) < 0.5) && (min_breakable_vs_broken(dd,1) == 1)
        min_breakable_vs_broken(dd,1) = epsilons(ii);
    elseif ~class1 && (pi_UUs_a(ii) >= 0.5) && (min_breakable_vs_broken(dd,1) == 1)
        min_breakable_vs_broken(dd,1) = epsilons(ii);
    end
    
end

disp('DONE!!!')


%% Plotting results
boundary = 0.5*ones(1,length(epsilons));

fig = figure;
plot(epsilons,attacks_GPFGS.probs(2,:),'r','DisplayName','GPFGS attack')
hold on
plot(epsilons,boundary,'m','HandleVisibility','off')
hold on
%legend(legendplot,strcat('sampled w',n_samples))
if defense.analytical(2,1) < 0.5
    plot(epsilons,defense.analytical(2,:),'b','DisplayName','upper bound maximum')
    hold on
    plot(epsilons,defense.analytical_exact(2,:),'b--','DisplayName','lower bound maximum')
    hold off
    ylim([0 0.55])
    legend('Location','southeast')
else
    ylim([0.45 1.0])
    legend('Location','northeast')
    plot(epsilons,defense.analytical(2,:),'b','DisplayName','lower bound minimum')
    hold on
    plot(epsilons,defense.analytical_exact(2,:),'b--','DisplayName','upper bound minimum')
    hold off
end
title(strcat('Testpoint',num2str(Testpoints(1))))
xlabel('gamma')
ylabel('pi')

