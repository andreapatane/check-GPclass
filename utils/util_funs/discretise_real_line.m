function   a = discretise_real_line(N)
%a = logit_fun(linspace(0,1,N));
a = linspace(-4,4,N);
a(1) = -realmax;
a(end) = realmax;
end


function out = logit_fun(p)

out = log( p./(1-p)  );
end