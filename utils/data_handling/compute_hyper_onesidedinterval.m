function [x_L, x_U] = compute_hyper_onesidedinterval(epsilon,testPoint, pixel2Modify,constrain_2_one)


x_L = testPoint;
x_U = testPoint;

if isstring(constrain_2_one)
    disp('');
end
for ii = 1:length(pixel2Modify)
    currPixel = pixel2Modify(ii);
    if constrain_2_one
        x_L(currPixel) = max(x_L(currPixel),0); % I take care that it is still a valid pixel
        x_U(currPixel) = min(x_U(currPixel) + epsilon,1.0); % I take care that it is still a valid pixel
    else
        x_L(currPixel) = x_L(currPixel) ; 
        x_U(currPixel) = x_U(currPixel) + epsilon; 
    end
end




end