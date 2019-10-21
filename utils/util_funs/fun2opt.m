function out = fun2opt(hyp,infFun,meanfunc,covfunc,likfunc,X_train,y_train)
hyp_struct.cov = hyp;
out = gp(hyp_struct,infFun,meanfunc,covfunc,likfunc,X_train,y_train);

end
