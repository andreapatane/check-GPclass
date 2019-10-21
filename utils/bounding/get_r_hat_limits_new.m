function [r_L_hat,r_U_hat] = get_r_hat_limits_new(r_L,r_U,B,a_vec,x_L,x_U)
% r_hat = U'*r. Given rectangular bounds for r, I here compute
% corresponding enclosing rectangular bounds for r_hat.

global U


r_L_hat = zeros(size(r_L));
r_U_hat = zeros(size(r_L));



%opts.Algorithm = 'interior-point';
%[x_opt,lb,exitFlag] = linprog(b_lin_obj_vec,B,-a_vec,[],[],[x_L,r_L_hat],[x_U,r_U_hat],opts);



ctype = repmat('U',length(a_vec),1);
vartype = repmat('C',size(B,2),1);
f0 = zeros(length(x_L),1);
x_L = [x_L,r_L];
x_U = [x_U,r_U];


for ii = 1:length(r_L)
    ii
    u_i = U(:,ii);
    %neg_idxs = u_i < 0;
    %pos_idxs = u_i >= 0;
    %r_star_L(pos_idxs) =  r_L(pos_idxs);
    %r_star_L(neg_idxs) =  r_U(neg_idxs);
    %r_star_U(pos_idxs) =  r_U(pos_idxs);
    %r_star_U(neg_idxs) =  r_L(neg_idxs);
    %r_L_hat_old(ii) = u_i'*r_star_L;
    %r_U_hat_old(ii) = u_i'*r_star_U;
    
    f = [f0;u_i];
    
    
    opts = optimoptions('linprog');
    opts.Display = 'off';
    tic
    [~,r_L_hat(ii),~] = linprog(f,B,-a_vec,[],[],x_L,x_U,opts);
    toc
    
    
    
    params.presol = 1;
    params.lpsolver = 1;
    params.msglev =3;
    params.toldj = 10e-16;
    params.objll = -inf;
    tic
    [x, r_L_hat(ii), status, ~] = glpk(f, B, -a_vec, x_L, x_U, ctype, vartype, 1,params);
    toc
    
    
    status
    B*x + a_vec
    %[~,r_U_hat_temp,~] = linprog(-f,B,-a_vec,[],[],[x_L,r_L],[x_U,r_U],opts);
    [x, r_U_hat(ii), status, ~] = glpk(f, B, -a_vec, x_L, x_U, ctype, vartype, -1);
    status
    %r_U_hat(ii) = - r_U_hat_temp;
end






end