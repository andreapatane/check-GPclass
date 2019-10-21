function [z_l,z_u] = propagate_diff(x_l,x_u,y_l,y_u)
    [z_l,z_u] = propagate_sum(x_l,x_u,-y_u,-y_l);
end