function out = LimeForGP(testPoint,n_samples,dataset,params_for_gp_toolbox)

global post
global training_data



dim = max(size(testPoint));

data = generate_lime_samples(testPoint,dim,n_samples,dataset);

distances = compute_distances(testPoint,data,0.25);
disp(size(distances))

[~, ~, a, ~,lp] = gp(params_for_gp_toolbox.hyp, params_for_gp_toolbox.infFun, ...
                  params_for_gp_toolbox.meanfunc, params_for_gp_toolbox.covfunc, params_for_gp_toolbox.likfunc,...
                  training_data, post, data, ones(size(data,1), 1));
                                         
probas = exp(lp);
disp(size(probas))

%Augment matrix so that intercept is fitted as well
A = [ones(n_samples,1),data];
disp(size(A))

%next: Fit linear model with weights = distances
w = lscov(A,probas,distances);

SSE = (A*w-probas)'*diag(distances)*(A*w-probas);
MSE = SSE / n_samples;

out.intercept = w(1);
out.coefs = w(2:(dim+1));
out.MSE = MSE;


