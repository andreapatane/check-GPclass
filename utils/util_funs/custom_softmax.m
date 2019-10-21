function y = custom_softmax(x)
%Returns softmax for input x

%x input matrix x, dimension 1 equalling number of classes and dimension 2
%   number of samples to convert

[c,n] = size(x);
y = zeros(c,n);
expo = exp(x);
sexpo = sum(expo,1);

for ii = 1:c
    y(ii,:) = expo(ii,:)./sexpo;
end

end