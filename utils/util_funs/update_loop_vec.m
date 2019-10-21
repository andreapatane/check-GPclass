function  update_loop_vec(mu_l,mu_u,sigma_u)

global loop_vec2

%d = 0.5*(mu_u - mu_l);


loop_vec2 = [loop_vec2(1), mu_l - 5*sigma_u mu_u + 5*sigma_u,loop_vec2(end)];


end