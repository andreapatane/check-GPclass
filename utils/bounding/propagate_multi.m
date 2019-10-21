
function [z_l,z_u] = propagate_multi(x_l,x_u,y_l,y_u)
values = [x_l*y_l,x_l*y_u,x_u*y_l,x_u*y_u];
z_l = min(values);
z_u = max(values);
end