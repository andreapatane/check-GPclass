function [yminpi, KpluWinv] = extract_values_multi_class_gpc(gp_model)
%Takes structure in which relevant information of multi class gp model with
%Laplace approximation are saved by multi-class-gpc package and extracts
%the items needed for calculation of mean (i.e. (y-pi)) and covariance
%(i.e. (K+W^{-1})^{-1}) of the predictive normal distribution.

%   gp_model   multi class gp laplace model structure as saved by the
%              multi-class-gpc package

K = gp_model.K;
W = gp_model.W;
y = gp_model.y;
Pi = gp_model.Pi;

n = length(y);
c = length(Pi)/length(y);

[ Ybin ] = full(ind2vec(y'))';
Y=reshape(Ybin,c*n,1);
yminpi = Y-Pi;

KpluWinv = inv(K+inv(W+10e-8*eye(size(W))));

end
