function [z_l,z_u] = propagate_div(x_l,x_u,y_l,y_u)
assert(y_l*y_u > 0)

if y_l > 0
    [z_l,z_u] = propagate_multi(x_l,x_u,1/y_l,1/y_u);
else
    [z_l,z_u] = propagate_multi(x_l,x_u,1/y_u,1/y_l);
end

end