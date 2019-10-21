function out = merging_1d_bounds(mu_f_u,mu_f_l,sigma_f_u,sigma_f_l,rect,mode)


dims = length(mu_f_l);
out = 1;
if strcmp(mode,'min')
    mode = 'inf';
elseif strcmp(mode,'max')
    mode = 'sup';
end
for ii = 1:dims
    out = out*compute_pi_i(mu_f_l(ii),mu_f_u(ii),sigma_f_l(ii),sigma_f_u(ii),rect(1,ii),rect(2,ii),mode);
end




end