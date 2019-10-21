function [r_L_hat,r_U_hat] = get_r_hat_limits_old(r_L,r_U,B,a_vec,x_L,x_U)


global U

r_L_hat = zeros(size(r_L));
r_U_hat = zeros(size(r_L));



m = length(x_L);
n = length(r_L);


opts = optimoptions('linprog');
opts.Display = 'off';
opts.Algorithm = 'interior-point-legacy';
opts.Preprocess = 'none';
opts.OptimalityTolerance = 1e-4;
opts.ConstraintTolerance = 1e-3;
opts.MaxIterations = 10000;
%nn = 1;
%a_vec_hat = - a_vec - B(:,1:m)*x_L' - B(:,(m+1):end)*r_L';
%B(:,1:m) = B(:,1:m).*((x_U - x_L)/nn);
%B(:,(m+1):end) = B(:,(m+1):end).*((r_U - r_L)/nn);

%aux = 0;

ps = parallel.Settings;
ps.Pool.AutoCreate = false;

parfor ii = 1:n
    %ii
    u_i = U(:,ii);
    %neg_idxs = u_i < 0;
    %pos_idxs = u_i >= 0;
    %r_star_L(pos_idxs) =  r_L(pos_idxs);
    %r_star_L(neg_idxs) =  r_U(neg_idxs);
    %r_star_U(pos_idxs) =  r_U(pos_idxs);
    %r_star_U(neg_idxs) =  r_L(neg_idxs);
    %r_L_hat_old(ii) = u_i'*r_star_L;
    %r_U_hat_old(ii) = u_i'*r_star_U;
    
    f = [zeros(m,1);u_i];
    
    %tic
    %[~,idxs] = maxk(abs(u_i),500);
    %try
    %idxs = [2*idxs;2*idxs-1];
 
    %f_hat = [zeros(m,1);u_i.*((r_U - r_L)/nn)'];
    [~,r_L_hat_hat,~] = linprog(f,B,- a_vec,[],[],[x_L,r_L],[x_U,r_U],opts);
    r_L_hat(ii) = r_L_hat_hat;
    
    
    %status
    [~,r_U_hat_temp,~] = linprog(-f,B,-a_vec,[],[],[x_L,r_L],[x_U,r_U],opts);
    %status
    r_U_hat(ii) = - r_U_hat_temp;

end


end