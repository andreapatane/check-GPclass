function [gp_trained_params,KWii,Kstarstar,a] = get_multiclass_gp_params(gp,X_train,y_train,X_test,y_test)

[n, ~] = size(X_train);
[nt,nout] = size(y_test);

gp_trained_params.kernel = 'sqe';
gp_trained_params.mle = false;
gp_trained_params.sigma = gp.cf{1, 1}.magnSigma2;
gp_trained_params.theta_vec = 0.5*(1./gp.cf{1, 1}.lengthScale.^2);
gp_trained_params.pred_fun = gp.lik.fh.predy;
gp_trained_params.lik = gp.lik;
gp_trained_params.nout = nout;
[~,~,~,p] = gpla_e(gp_pak(gp),gp,X_train,y_train,'z',[]);
[f,~,a,~,~] = deal(p.f,p.L,p.a,p.La2,p.p);

f2 = reshape(f,n,nout);
a = reshape(a,n,nout); %This is (y-pi), i.e. what you need to multiply with k* to get latent mean 
K = zeros(n,n,nout);
for i1 = 1:nout
    K(:,:,i1) = getKernel(X_train,X_train,gp_trained_params);
end
B = [];
for i1 = 1:nout
    B = blkdiag(B,K(:,:,i1));
end
K = B;



test_inputs = X_test;
test_outputs = y_test;
%[nt] = size(test_outputs,1);

%meanvec = zeros(nt,nout);
%Kstar = zeros(nt,n,nout);
%Kstarstar = zeros(nout,nout,nt);
%covmat = zeros(nout,nout,nt);

%for i1=1:nout
%    predcf{i1} = [];
%end


% _predictive covariance_
[pi2_vec,pi2_mat] = gp.lik.fh.llg2(gp.lik,y_train,f2,'latent',[]);
W = diag(pi2_vec) - pi2_mat*pi2_mat';
%W_inv = inv(W);
%if W_inv(1,1) == Inf
%    disp('W not invertible, added noise to diagonal')
W_inv = inv(W+10e-10*eye(size(W)));
%W_inv = inv_chol(chol(W+10e-10*eye(size(W)) ,'upper'));
%end
KWii = inv(K+W_inv ); %This is (K+W^{-1})^{-1}


disp('Predicting test points using manually computed latent mean and covariance')
aux = tic;
[proba,Kstarstar,meanvec,covmat] = gp_multiclass_manual_prediction(X_test,X_train,gp_trained_params,a,KWii);
toc(aux)

%% Predict with package
% make the prediction for test points
disp('Predicting test points using GPstuff package prediction function')
aux = tic;
[Eft, Varft, lpyt] = gp_pred(gp, X_train, y_train, X_test, 'yt', ones(size(y_test)));

Eft = reshape(Eft, size(X_test,1), size(y_test,2));
lpyt = reshape(lpyt, size(X_test,1), size(y_test,2));
toc(aux)
disp('done')


%%  Evaluate results

% calculate the percentage of misclassified points
%y_predict = exp(lpyt)==repmat(max(exp(lpyt),[],2),1,size(exp(lpyt),2));
%y_predict = 1*y_predict;
%missed = (sum(sum(abs(y_predict-y_test)))/2)/size(y_test,1)

% calculate the percentage of misclassified points
ym_predict = (exp(proba) == repmat(max(exp(proba),[],2),1,size(proba,2)));
ym_predict = 1*ym_predict;
missedm = (sum(sum(abs(ym_predict-y_test)))/2)/size(y_test,1);
disp('Test set accuracy: ')
disp(1 - missedm)

%if plottingFlag
%    scrsz = get(0,'ScreenSize');
%    fig3 = figure(3);
%    set(fig3, 'name', 'The confusion matrix for GPstuff', 'Position',[1000 scrsz(4) 500 400]);
%    plotconfusion(y_test', y_predict');
%    title('Exact GPstuff')
%end
%
%if plottingFlag
%    scrsz = get(0,'ScreenSize');
%    fig4 = figure(4);
%    set(fig4, 'name', 'The confusion matrix for manual GPstuff', 'Position',[1000 scrsz(4) 500 400]);
%    plotconfusion(y_test', ym_predict');
%    title('Manual GPstuff')
%end
            
%% Compare manual to package

disp('difference means self-computed and computed by package =')
disp(max(sum(abs(meanvec-Eft))))
disp('maximum total absolute difference of any testpoint between covariance self-computed and computed by package  =')
disp(max(sum(sum(abs(covmat-Varft)))))
disp('Difference in predicted probabilities on average')
disp((sum(abs(exp(lpyt)-exp(proba))))/nt)
%disp('Number of test points assigned to different class')
%disp((max(sum(abs(ym_predict-y_predict)))))
end
