function [c_x_l,c_x_u] = expand_domain_variables(reg,pixels2modify,x_L)
%expands compact representations of region to full representations

c_x_l = x_L;
c_x_u = x_L;
c_x_l(pixels2modify) = reg(1,:);
c_x_u(pixels2modify) = reg(2,:);


end
