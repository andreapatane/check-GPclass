%function [lb,x_mu_lb] = compute_lower_bound_mu_sqe(y_i_vec,x_L,x_U,theta_vec,sigma_prior,training_data)
function [lb,x_mu_lb] = compute_lower_bound_mu_sqe(y_i_vec,x_L,x_U,theta_vec,sigma_prior,z_i_L_vec,z_i_U_vec)

global training_data
global theta_vec_train_squared

%scaling output with a priori variance
y_i_vec = y_i_vec*sigma_prior;

n = length(theta_vec_train_squared);

%debugging mu formula
%mu = get_actual_prediction(training_data,testPoint,n,theta_vec,y_i_vec);

m = size(training_data,2);
a_i_sum = 0;
b_i_vec = zeros(1,n);
for ii = 1:n
    y_i = y_i_vec(ii);
    %x_i = training_data(ii,:);
    %[z_i_L, z_i_U] = compute_z_interval(x_i,x_L,x_U,theta_vec);
    z_i_L = z_i_L_vec(ii);
    z_i_U = z_i_U_vec(ii);
    if z_i_L >= z_i_U
        z_i_L = z_i_L - 1.0e-12;
    end
    %[a_i,b_i] = compute_linear_under_approx(y_i,z_i_L,z_i_U);
    %Inlning stuff for computational reasons
    if y_i >= 0
        z_i_M = 0.5*(z_i_L + z_i_U);
        a_i = (1 + z_i_M)*y_i*exp(-z_i_M);
        b_i = -y_i*exp(-z_i_M);
    else
        a_i = y_i * ( exp(-z_i_L)  - z_i_L * (  exp(-z_i_L) -  exp(-z_i_U) )/( z_i_L - z_i_U   )  );
        b_i = y_i * (  exp(-z_i_L) -  exp(-z_i_U) )/( z_i_L - z_i_U   );
    end
    %End of Inlning
    b_i_vec(ii) = b_i;
    a_i_sum = a_i_sum + a_i;
end


H = 2*sum(b_i_vec)*theta_vec;

f = zeros(m,1);
for jj = 1:m
    f(jj) = - 2 * theta_vec(jj)*b_i_vec*training_data(:,jj);
end
C = 0;
for ii = 1:n
    %for jj = 1:m
    %    C = C + theta_vec(jj) * b_i_vec(ii) *training_data(ii,jj)^2;
    %end
    C = C + b_i_vec(ii) * theta_vec_train_squared(ii);
end

[x_mu_lb, f_val] = separate_quadprog(H,f,x_L,x_U);
x_mu_lb  = x_mu_lb';
lb = f_val + a_i_sum + C;

%ub = get_actual_prediction(training_data,x_mu_lb,theta_vec,y_i_vec);
%lb = f_val;
% 
% if debug
%     %debugging
%     pix = linspace(x_L(103),x_U(103),10);
%     out_gp_debug = zeros(size(pix));
%     out_quadratic_debug = zeros(size(pix));
%     out_fu_under_debug = zeros(size(pix));
% 
%     for ii = 1:length(pix)
%         x_star(103) = pix(ii);
%         out_gp_debug(ii) = get_actual_prediction(training_data,x_star',n,theta_vec,y_i_vec);
%         out_quadratic_debug(ii) = 0.5*x_star'*H*x_star + f'* x_star + a_i_sum + C;
%         out_fu_under_debug(ii) = f_under(a_i_sum,b_i_vec,theta_vec,training_data,x_star');
%     end
%     figure
%     hold on
%     plot(pix,out_gp_debug)
%     plot(pix,lb*ones(size(pix)))
%     plot(pix,out_quadratic_debug)
%     plot(pix,f_val*ones(size(pix)))
%     plot(pix,out_fu_under_debug)
% 
%     legend({'gp','lb','quadratic','f_val','f_under'})
% end


end


function mu = get_actual_prediction(training_data,testPoint,theta_vec,y_i_vec)
mu = 0;
for ii = 1:size(training_data,1)
    mu = mu +  exp(- dot(theta_vec, (training_data(ii,:) - testPoint).^2))*y_i_vec(ii) ;
end

end


