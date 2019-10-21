function X = standardise(X)

mean_x = mean(X);
std_x = std(X);

for ii = 1:size(X,2)
   X(:,ii) =  (X(:,ii) - mean_x(ii))/std_x(ii);
end

end