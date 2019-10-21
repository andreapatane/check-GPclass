%function [sigma_lb,x_sigma_lb] = compute_lower_bound_sigma_sqe(R_inv,x_L,x_U,theta_vec,sigma_prior,training_data)
function [sigma_lb,x_sigma_lb] = compute_lower_bound_sigma_sqe(x_L,x_U,theta_vec,sigma_prior,z_i_L_vec,z_i_U_vec)

global training_data
global R_inv
global theta_vec_train_squared

%R_inv = sigma_prior * R_inv;

n = size(training_data,1);
m = size(training_data,2);

%z_i_L_vec = zeros(1,n);
%z_i_U_vec = zeros(1,n);
                                            
%B_il =  zeros(n,n);
B_sum = zeros(1,n);
a_il_sum = 0;

%for ii = 1:n
%    [z_i_L_vec(ii),z_i_U_vec(ii)] = compute_z_interval(training_data(ii,:),x_L,x_U,theta_vec);
%end                                           

for ii = 1:n
    
    for ll = 1:ii
        
        z_il_L = z_i_L_vec(ii) + z_i_L_vec(ll);
        z_il_U = z_i_U_vec(ii) + z_i_U_vec(ll);
        
        %[a_il,B_il(ii,ll)] = compute_linear_under_approx(-R_inv(ii,ll),z_il_L,z_il_U);
        %Inlning stuff for computational reasons
        %y_i = (- R_inv(ii,ll));
        if (- R_inv(ii,ll)) >= 0
            z_i_M = 0.5*(z_il_L + z_il_U);
            a_il = (1 + z_i_M)*(- R_inv(ii,ll))*exp(-z_i_M);
            %B_il(ii,ll) = -(- R_inv(ii,ll))*exp(-z_i_M);
            B_il_var = -(- R_inv(ii,ll))*exp(-z_i_M);
        else
            a_il = (- R_inv(ii,ll)) * ( exp(-z_il_L)  - z_il_L * (  exp(-z_il_L) -  exp(-z_il_U) )/( z_il_L - z_il_U   )  );
            %B_il(ii,ll) = (- R_inv(ii,ll)) * (  exp(-z_il_L) -  exp(-z_il_U) )/( z_il_L - z_il_U   );
            B_il_var = (- R_inv(ii,ll)) * (  exp(-z_il_L) -  exp(-z_il_U) )/( z_il_L - z_il_U   );
        end
        %End of Inlning
        
        %plot_debug(z_il_L,z_il_U,a_il,b_il,R_inv(ii,ll))
        a_il = - a_il;
        %B_il(ii,ll) = - B_il(ii,ll);
        B_il_var = - B_il_var;
        B_sum(ii) = B_sum(ii) + B_il_var;

        if ll < ii
            a_il = 2*a_il;
            %B_il(ll,ii) = B_il(ii,ll);
            B_sum(ll) = B_sum(ll) + B_il_var;
        end
        
        a_il_sum = a_il_sum + a_il;
    end
    
    
end

%B_sum = sum(B_il,2);

%C = 0;
%for ii = 1:n
%    C = C + B_sum(ii) * dot(theta_vec,training_data(ii,:).^2);
%end
%C = 2*C;
C = 2 * B_sum * theta_vec_train_squared;

H = 4*sum(B_sum) * theta_vec;

f = zeros(m,1);
for  jj = 1:m
    f(jj) = -4*theta_vec(jj) * dot(B_sum,training_data(:,jj));
end


[x_star, f_val] = separate_quadprog(-H',-f,x_L,x_U);
f_val = - f_val;
f_val = f_val + a_il_sum + C;
ub = f_val;

x_sigma_lb = x_star';

sigma_lb = sigma_prior*(1 - ub);

%clipping lower bound to zero.
sigma_lb = max(sigma_lb,0);
end