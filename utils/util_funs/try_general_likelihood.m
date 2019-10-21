clear all
close all
clc

mu_l = -7.7819;
mu_u = -7.7491;
sigma_l = 6.1728;
sigma_u = 8.5655;
N = 1000;
mode = 'max';

%probit_likelihood = @(t) (0.5*(1 + erf(t./sqrt(2))));


[a,b] = compute_pi_extreme(@normcdf,mu_l,mu_u,sigma_l,sigma_u,-100,100,N,mode);
sum(a)



probit_likelihood_analytical_solution(mu_l,sigma_l,1)
probit_likelihood_analytical_solution(mu_l,sigma_u,1)
probit_likelihood_analytical_solution(mu_u,sigma_l,1)
probit_likelihood_analytical_solution(mu_u,sigma_u,1)
