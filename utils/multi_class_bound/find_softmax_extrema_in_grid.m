function [multi_grid_mins,multi_grid_maxs] = find_softmax_extrema_in_grid(classIdx)
global multi_grid
multi_grid_mins = zeros(size(multi_grid));
multi_grid_maxs = zeros(size(multi_grid));

for ii = 1:length(multi_grid)
    lbs = multi_grid{ii}(1,:);
    ubs = multi_grid{ii}(2,:);
    
    min_point = ubs;
    min_point(classIdx) = lbs(classIdx);
    
    max_point = lbs;
    max_point(classIdx) = ubs(classIdx);
    multi_grid_mins(ii) = softmax_lik(min_point,classIdx); 
    multi_grid_maxs(ii) = softmax_lik(max_point,classIdx);
    
end




end