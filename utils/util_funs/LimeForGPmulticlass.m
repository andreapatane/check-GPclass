function out = LimeForGPmulticlass(testPoint,n_samples,dataset,params_for_gp_toolbox)


global training_data
global test_data



a = params_for_gp_toolbox.a;
KWii = params_for_gp_toolbox.KWii;


dim = max(size(testPoint));

coefs = zeros(dim,3);

data = generate_lime_samples(testPoint,dim,n_samples,dataset);

distances = compute_distances(testPoint,data,0.25);
disp(size(distances))

[proba,Kstarstar,meanvec,covmat] = gp_multiclass_manual_prediction(data,training_data,params_for_gp_toolbox,a,KWii)
                                         
disp(size(proba))

maski = testPoint > 0;

data = data(:,maski);

%Augment matrix so that intercept is fitted as well
A = [ones(n_samples,1),data];
disp(size(A))

%next: Fit linear model with weights = distances
w = lscov(A,proba,distances);

temp = w(2:end,:);
coefs(maski,:) = temp;

SSE = diag((A*w-proba)'*diag(distances)*(A*w-proba));
MSE = SSE / n_samples;

out.intercept = w(1,:);
out.coefs = coefs;
out.MSE = MSE;

end