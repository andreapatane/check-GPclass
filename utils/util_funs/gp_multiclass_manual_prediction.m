function [proba,Kstarstar,meanvec,covmat] = gp_multiclass_manual_prediction(X_test,X_train,gp_trained_params,a,KWii)

nout = gp_trained_params.nout;
nt = size(X_test,1);
n = size(X_train,1);
meanvec = zeros(nt,nout);
Kstar = zeros(nt,n,nout);
Kstarstar = zeros(nout,nout,nt);
covmat = zeros(nout,nout,nt);


% Computing mean
for i1=1:nout
    Kstar(:,:,i1) = getKernel(X_test,X_train,gp_trained_params);
    Kstarstar(i1,i1,:) = diag(getKernel(X_test,X_test,gp_trained_params));
    meanvec(:,i1) = Kstar(:,:,i1)*a(:,i1);
end

%adding jitter to Kstarstar
for ii = 1:nt
    Kstarstar(:,:,ii) = Kstarstar(:,:,ii)+0.01*eye(nout,nout);
end

%Computing variance
for ii = 1:nt
    Qstar = [];
    for jj = 1:nout
        Qstar = blkdiag(Qstar,Kstar(ii,:,jj));
    end
    %covmat(:,:,ii) = Kstarstar(:,:,ii)+0.01*eye(nout,nout) - Qstar*KWii*Qstar';
    covmat(:,:,ii) = Kstarstar(:,:,ii) - Qstar*KWii*Qstar';
end

%Correct for asymmetries introduced by adding noise
symmetrydiffs = zeros(nt,3);
for jj = 1:nt
    testpoint = jj;
    symmetrydiffs(jj,1) = covmat(2,1,testpoint) - covmat(1,2,testpoint);
    symmetrydiffs(jj,2) = covmat(3,1,testpoint) - covmat(1,3,testpoint);
    symmetrydiffs(jj,3) = covmat(3,2,testpoint) - covmat(2,3,testpoint);
end
if max(max(symmetrydiffs) < 0.1)
    for jj = 1:nt
        testpoint = jj;
        temp21 = 0.5*(covmat(2,1,testpoint) + covmat(1,2,testpoint));
        temp31 = 0.5*(covmat(3,1,testpoint) + covmat(1,3,testpoint));
        temp32 = 0.5*(covmat(3,2,testpoint) + covmat(2,3,testpoint));
        covmat(2,1,testpoint) = temp21;
        covmat(1,2,testpoint) = temp21;
        covmat(3,1,testpoint) = temp31;
        covmat(1,3,testpoint) = temp31;
        covmat(3,2,testpoint) = temp32;
        covmat(2,3,testpoint) = temp32;
    end
end

% Calculate predictive probabilities using package but our mean and covariance_
%proba = gp.lik.fh.predy(gp.lik, meanvec, covmat, ones(nt,nout), []);
proba = gp_trained_params.pred_fun(gp_trained_params.lik, meanvec, covmat, ones(nt,nout), []);
proba = reshape(proba, nt, nout);







end