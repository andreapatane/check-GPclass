function pi_i = compute_pi_i(mu_l,mu_u,sigma_l,sigma_u,a_i,b_i,mode)
%Computes the extreme values (inf or sup) of a gaussian pdf integrated over
%interval [a_i,b_i] with possible \mu and \sigma values between [mu_l,mu_u]
%and [sigma_l,sigma_u].

%   mu_l    lower bound for mean of Gaussian pdf
%   mu_u    upper bound for mean of Gaussian pdf
%   sigma_l lower bound for std of Gaussian pdf
%   sigma_u upper bound for std of Gaussian pdf
%   a_i     lower bound of interval that is being integrated over
%   b_i     upper bound of interval that is being integrated over
%   mode    'sup' or 'inf'





%check that interval borders are in right order (smaller value first):
%assert(mu_l <= mu_u)
%assert(sigma_l <= sigma_u)
%assert(a_i < b_i)
%assert(sigma_l >= 0)

m = 0.5*(a_i+b_i);

switch mode
    %Minimum Case:
    case 'inf'
        
        if m >= mu_u
            mu_s = mu_l;
        elseif m <= mu_l
            mu_s = mu_u;
        else
            aa = abs(mu_l - m);
            bb = abs(mu_u - m);
            if aa >= bb
                mu_s = mu_l;
            else
                mu_s = mu_u;
            end
        end
        
        p_l = integral_solution(mu_s,sigma_l,a_i,b_i);
        p_u = integral_solution(mu_s,sigma_u,a_i,b_i);
        pi_i = min(p_l,p_u);
    
    %Maximum Case:
    case 'sup'
        
        
        if m >= mu_u
            mu_s = mu_u;
        elseif m <= mu_l
            mu_s = mu_l;    
        else
            mu_s = m;
        end
        
        if (mu_s >= a_i) && (mu_s <= b_i)
            sigma_s = sigma_l;
        else
            c = ((mu_s - a_i).^2-(mu_s - b_i).^2)/(2*log((mu_s - a_i)/(mu_s - b_i)));
            if isnan(c)
                c = 0;
            end
            if c >= sigma_u
                sigma_s = sigma_u;
            elseif c <= sigma_l
                sigma_s = sigma_l;                
            else
                sigma_s = c;
            end
        end
        pi_i = integral_solution(mu_s,sigma_s,a_i,b_i);
end


end


function out = integral_solution(mu,sigma,a,b)
out = 0.5*(   erf((mu-a)/sqrt(sigma*2))-erf((mu-b)/sqrt(sigma*2))   );
%if isnan(out)
%    disp('')
%end
end