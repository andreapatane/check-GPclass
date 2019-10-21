function deri = sigmaderivative(mu,v,sigma,k)

r = 1.0./sqrt(v.^2+sigma.^2);
kvr = k.*v.*r;
muv = mu./v;

a = mu.*r.*sigma.^4*normpdf(kvr).*(normcdf(k+sigma.*muv.*r)-normcdf(sigma.*muv.*r));
b = sigma.^4.*kvr.*normpdf(kvr).*normcdf(kvr+muv);
c = sigma.^3.*(sigma.^2+v.^3).*r.^2.*(1.0./sqrt(2*pi)*normpdf(muv) - normpdf(kvr).*normpdf(k.*sigma.*r + muv));
d = kvr.*normpdf(kvr).*normcdf(muv + kvr);


deri = -a-b+c-d;


end


