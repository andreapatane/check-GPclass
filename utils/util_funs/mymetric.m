function [numb] = mymetric(y, pred, type)

if nargin<3, error('Specify metric type'); end

switch type 
  case 'CE'
    if sum(y==-1)>0, y = (y+1)./2; end  %set class labels to 0 and 1
    numb = -sum(y.*log(pred) + (1-y).*log(1-pred));

  case 'MSE'
    numb = (y-pred)'*(y-pred);

  case 'Acc'
    if sum(y==-1)>0, y = (y+1)./2; end %set class labels to 0 and 1
    predones = (pred>0.5);
    numb = sum(predones==y)/numel(y);
        
end
end