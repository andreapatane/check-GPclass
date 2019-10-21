function [z_i_L_vec,z_i_U_vec] = pre_compute_z_intervals(x_L,x_U,theta_vec)

global training_data

n = size(training_data,1);
z_i_L_vec = zeros(1,n);
z_i_U_vec = zeros(1,n);
                                            
for ii = 1:n
    [z_i_L_vec(ii),z_i_U_vec(ii)] = compute_z_interval(training_data(ii,:),x_L,x_U,theta_vec);
end                                           

end