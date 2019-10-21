function X = normalise(X)

max_x = max(X);
min_x = min(X);

for ii = 1:size(X,2)
   %X(:,ii) =  (X(:,ii) - mean_x(ii))/std_x(ii);
   X(:,ii) =  (X(:,ii) - min_x(ii))/(max_x(ii) - min_x(ii));
end

end