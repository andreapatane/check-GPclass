function [z_l,z_u] = propagate_sum(x_l,x_u,y_l,y_u)
z_l = x_l + y_l;
z_u = x_u + y_u;
end