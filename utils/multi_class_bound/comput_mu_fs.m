function [mu_f_u,mu_f_l] = comput_mu_fs(mu_l,mu_u,sigma_l,sigma_u,rect)


dims = size(rect,2);
mu_f_u = zeros(1,dims);
mu_f_l = zeros(1,dims);

%%%%%%%%
%Class 3
mu_f_u(3) = mu_u(3);
mu_f_l(3) = mu_l(3);

%%%%%%%%
%Class 2
f3_l = rect(1,3);
f3_u = rect(2,3);

%Mean:
%defining z_3 = f_3 - mu_3
[z3_l,z3_u] = propagate_diff(f3_l,f3_u,mu_l(3),mu_u(3));
%sigma23*z3
[temp_l,temp_u] = propagate_multi(sigma_l(2,3),sigma_u(2,3),z3_l,z3_u);
%(sigma23*z3)/sigma33
[temp_l,temp_u] = propagate_div(temp_l,temp_u,sigma_l(3,3),sigma_u(3,3));
%
[mu_f_l(2),mu_f_u(2)] = propagate_diff(mu_l(2),mu_u(2),temp_l,temp_u);


%%%%%%%%
%Class 1

%Mean:
f2_l = rect(1,2);
f2_u = rect(2,2);

%z2 = f2 - mu2
[z2_l,z2_u] = propagate_diff(f2_l,f2_u,mu_l(2),mu_u(2));

%sigma22*sigma33
[fact1_l,fact1_u] = propagate_multi(sigma_l(2,2),sigma_u(2,2),sigma_l(3,3),sigma_u(3,3));
%sigma23^2
[fact2_l,fact2_u] = propagate_multi(sigma_l(2,3),sigma_u(2,3),sigma_l(2,3),sigma_u(2,3));
%det = sigma22*sigma33 - sigma23^2
[det_l,det_u] = propagate_diff(fact1_l,fact1_u,fact2_l,fact2_u);

%sigma12*sigma33
[fact1_l,fact1_u] = propagate_multi(sigma_l(1,2),sigma_u(1,2),sigma_l(3,3),sigma_u(3,3));
%sigma13*sigma23
[fact2_l,fact2_u] = propagate_multi(sigma_l(1,3),sigma_u(1,3),sigma_l(2,3),sigma_u(2,3));
%sigma12*sigma33 - sigma13*sigma23 
[cross1_l,cross1_u] = propagate_diff(fact1_l,fact1_u,fact2_l,fact2_u);
%summand1 = z2 * (sigma12*sigma33 - sigma13*sigma23 )
[summand1_l,summand1_u] = propagate_multi(z2_l,z2_u,cross1_l,cross1_u);

%sigma13*sigma22
[fact1_l,fact1_u] = propagate_multi(sigma_l(1,3),sigma_u(1,3),sigma_l(2,2),sigma_u(2,2));
%sigma12*sigma23
[fact2_l,fact2_u] = propagate_multi(sigma_l(1,2),sigma_u(1,2),sigma_l(2,3),sigma_u(2,3));
%sigma13*sigma22 - sigma12*sigma23 
[cross2_l,cross2_u] = propagate_diff(fact1_l,fact1_u,fact2_l,fact2_u);
%summand2 = z3 * (sigma13*sigma22 - sigma12*sigma23 )
[summand2_l,summand2_u] = propagate_multi(z3_l,z3_u,cross2_l,cross2_u);
%z2 * (sigma12*sigma33 - sigma13*sigma23 ) + z3 * (sigma13*sigma22 - sigma12*sigma23 )
[temp_l,temp_u] = propagate_sum(summand1_l,summand1_u,summand2_l,summand2_u);
%dividing for determinant
[temp_l,temp_u] = propagate_div(temp_l,temp_u,det_l,det_u);
%Final expression
[mu_f_l(1),mu_f_u(1)] = propagate_diff(mu_l(1),mu_u(1),temp_l,temp_u);


end