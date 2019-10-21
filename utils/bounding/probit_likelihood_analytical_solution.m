function out = probit_likelihood_analytical_solution(mu,var,scale)
if var < 0
    disp('Warning: passed negative variance to probit function')
    disp(var)
    var = 0;
end


out =  0.5*(1 +erf(mu ./ sqrt(2*(scale^(-2) + var)))) ;

end