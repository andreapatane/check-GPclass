function [pi_bar] = compute_pi_extreme(likelihood,mu_l,mu_u,sigma_l,sigma_u,a_L,b_U,N,mode)

global loop_vec2
%TO DO: WRITE COMMENTS HERE

%check that interval borders are in right order (smaller value first):
%assert(a_L < b_U)
%assert(mu_l <= mu_u)
%assert(sigma_l <= sigma_u)
%assert(sigma_l >= 0)


%loop_vec2 = [-realmax,a_L:((b_U-a_L)/N):b_U,realmax];
%loop_vec2 = discretise_real_line(N);
pi_bar = 0.0;

fun = @(x) likelihood(x);

switch mode
    %Minimum Case:
    case 'min'
        
        for ii = 1:(length(loop_vec2)-1)
            a_i = loop_vec2(ii);
            b_i = loop_vec2(ii+1);
            m = 0.5*(a_i+b_i);
            l_i = fun(a_i);
            %pi_bar = pi_bar + l_i*compute_pi_i(mu_l,mu_u,sigma_l,sigma_u,a_i,b_i,'inf');
            %%Inlining function
            
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
            pi_bar = pi_bar + l_i * min(p_l,p_u);
            %%End function inlining
            
        end
     
    case 'max'
        
        for ii = 1:(length(loop_vec2)-1)
            a_i = loop_vec2(ii);
            b_i = loop_vec2(ii+1);
            u_i = fun(b_i);
            %pi_bar = pi_bar + u_i*compute_pi_i(mu_l,mu_u,sigma_l,sigma_u,a_i,b_i,'sup');
            %Start function inline
            m = 0.5*(a_i+b_i);
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
            pi_bar = pi_bar + u_i*integral_solution(mu_s,sigma_s,a_i,b_i);
            %End function inline
        end

end
        
end


function out = integral_solution(mu,sigma,a,b)
out = 0.5*(   erf((mu-a)/sqrt(sigma*2))-erf((mu-b)/sqrt(sigma*2))   );
end
