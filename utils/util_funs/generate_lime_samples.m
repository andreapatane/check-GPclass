function samples = generate_lime_samples(testPoint,dim,n_samples,datatype)



if datatype == 'img'
    mask = randi([0,1],n_samples,dim);
    target = repmat(testPoint,n_samples,1);
    mask(1,:) = ones(1,dim);
    samples = target .* mask;
elseif datatype == 'reg'
    mask = randn(n_samples,dim);
    target = repmat(testPoint,n_samples,1);
    mask(1,:) = zeros(1,dim);
    samples = target+mask;
else
    disp('sampling not implemented for type of data yet')
end

end
    


