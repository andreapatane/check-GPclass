function distances = compute_distances(testPoint,data,kernel_width)

n_s = length(data(:,1));

distances = zeros(n_s,1);

for ii = 1:n_s
    similarity = pdist([testPoint;data(ii,:)],'cosine');
    distances(ii) = max(sqrt(exp(-(similarity^2) / kernel_width^2)),0.01);
end

end
    
    
