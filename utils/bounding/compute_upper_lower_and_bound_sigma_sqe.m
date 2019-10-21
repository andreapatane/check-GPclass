function [sigma_l,x_sigma_l,sigma_u,x_sigma_u] = compute_upper_lower_and_bound_sigma_sqe(x_L,x_U,theta_vec,sigma_prior,z_i_L_vec,...
    z_i_U_vec,bound_comp_opts,mu_sigma_bounds,low_or_up,offset_row,offset_cols)

if nargin < 11
    offset_cols = 0;
    if nargin < 10
        offset_row = 0;
        if nargin < 9
            low_or_up = 'both';
        end
    end
end



flagComputeUB = strcmp(low_or_up,'upper') || strcmp(low_or_up,'both');
flagComputeLB = strcmp(low_or_up,'lower') || strcmp(low_or_up,'both');

sigma_l = [];
x_sigma_l = [];
sigma_u = [];
x_sigma_u = [];


global training_data
global R_inv
global theta_vec_train_squared

%R_inv = sigma_prior * R_inv;

n = size(training_data,1);
m = size(training_data,2);

%z_i_L_vec = zeros(1,n);
%z_i_U_vec = zeros(1,n);
                                            
%B_il_U =  zeros(n,n);
B_sum_U = zeros(1,n);
B_sum_L = zeros(1,n);
a_il_sum_U = 0;

%B_il_L =  zeros(n,n);
a_il_sum_L = 0;

%for ii = 1:n
%    [z_i_L_vec(ii),z_i_U_vec(ii)] = compute_z_interval(training_data(ii,:),x_L,x_U,theta_vec);
%end                                           

for ii = 1:n
    
    r_inv_ii = R_inv(offset_row + ii, offset_cols + (1:ii));
    
    
    
    for ll = 1:ii
        
        z_il_L = z_i_L_vec(ii) + z_i_L_vec(ll);
        z_il_U = z_i_U_vec(ii) + z_i_U_vec(ll);
        z_i_M = 0.5*(z_il_L + z_il_U);
        
        if r_inv_ii(ll) >= 0 
            
            a_il_U = (1 + z_i_M)*r_inv_ii(ll)*exp(-z_i_M);
            B_il_U_var = -r_inv_ii(ll)*exp(-z_i_M);
            a_il_L = (- r_inv_ii(ll)) * ( exp(-z_il_L)  - z_il_L * (  exp(-z_il_L) -  exp(-z_il_U) )/( z_il_L - z_il_U   )  );
            B_il_L_var = (-r_inv_ii(ll)) * (  exp(-z_il_L) -  exp(-z_il_U) )/( z_il_L - z_il_U   );
            
        else
            a_il_U = r_inv_ii(ll) * ( exp(-z_il_L)  - z_il_L * (  exp(-z_il_L) -  exp(-z_il_U) )/( z_il_L - z_il_U   )  );
            B_il_U_var = r_inv_ii(ll) * (  exp(-z_il_L) -  exp(-z_il_U) )/( z_il_L - z_il_U   );
            a_il_L = (1 + z_i_M)*(- r_inv_ii(ll))*exp(-z_i_M);
            B_il_L_var = -(- r_inv_ii(ll))*exp(-z_i_M);
        end
        %End of Inlning
        %plot_debug(z_il_L,z_il_U,a_il,b_il,R_inv(ii,ll))
        a_il_L = - a_il_L;
        %B_il_L(ii,ll) = - B_il_L(ii,ll);
        B_il_L_var = - B_il_L_var;
        B_sum_U(ii) = B_sum_U(ii) + B_il_U_var;
        B_sum_L(ii) = B_sum_L(ii) + B_il_L_var;
        if ll < ii
            a_il_U = 2*a_il_U;
            B_sum_U(ll) = B_sum_U(ll) + B_il_U_var;
            a_il_L = 2*a_il_L;
            B_sum_L(ll) = B_sum_L(ll) + B_il_L_var;
        end
        a_il_sum_L = a_il_sum_L + a_il_L;
        a_il_sum_U = a_il_sum_U + a_il_U;
        
    end
    
    
end


C_U = 2 * B_sum_U * theta_vec_train_squared;


C_L = 2 * B_sum_L * theta_vec_train_squared;


H_U = 4*sum(B_sum_U) * theta_vec;
H_L = 4*sum(B_sum_L) * theta_vec;

f_U = zeros(m,1);
for  jj = 1:m
    f_U(jj) = -4*theta_vec(jj) * B_sum_U*training_data(:,jj);
end

f_L = zeros(m,1);
for  jj = 1:m
    f_L(jj) = -4*theta_vec(jj) * B_sum_L*training_data(:,jj);
end


if flagComputeLB
    [x_star_L, f_val_L] = separate_quadprog(-H_L',-f_L,x_L,x_U);
    f_val_L = - f_val_L;
    f_val_L = f_val_L + a_il_sum_L + C_L;
    ub = f_val_L;
    
    x_sigma_l = x_star_L;
    if offset_row == offset_cols
        sigma_l = sigma_prior*(1 - ub);
        sigma_l = max(sigma_l,0);
    else
        sigma_l = -sigma_prior*ub;
    end
end
%clipping lower bound to zero.



if flagComputeUB
    [x_star_U, f_val_U] = separate_quadprog(H_U',f_U,x_L,x_U);
    f_val_U = f_val_U + a_il_sum_U + C_U;
    lb = f_val_U;
    x_sigma_u = x_star_U;
        
    if offset_row == offset_cols
        sigma_u = sigma_prior*(1 - lb);
    else
        sigma_u = -sigma_prior*lb;
    end
    
        
    if ~isempty(sigma_l)
        sigma_u = max(sigma_l,sigma_u);  
    end
end





end


  
