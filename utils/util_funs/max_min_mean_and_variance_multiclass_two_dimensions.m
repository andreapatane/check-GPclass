function [min_mu,max_mu,min_var,max_var] = max_min_mean_and_variance_multiclass_two_dimensions(testPoint,bound_comp_opts,params_for_gp_toolbox,trainedSystem,KWii)

global training_data

pix2mod = find(bound_comp_opts.x_U > bound_comp_opts.x_L);
testPointInit = testPoint;

t_x = linspace(bound_comp_opts.x_L(pix2mod(1)),bound_comp_opts.x_U(pix2mod(1)),25);
t_y = linspace(bound_comp_opts.x_L(pix2mod(2)),bound_comp_opts.x_U(pix2mod(2)),25);

min_mu = inf(1,params_for_gp_toolbox.nout);
max_mu = -inf(1,params_for_gp_toolbox.nout);
min_var = inf(params_for_gp_toolbox.nout,params_for_gp_toolbox.nout);
max_var = -inf(params_for_gp_toolbox.nout,params_for_gp_toolbox.nout);
for ii = 1:length(t_x)
   for jj = 1:length(t_y)
       testPoint = testPointInit;
       testPoint(pix2mod) = [t_x(ii),t_y(jj)];
       [proba,Kstarstar,meanvec,covmat] = gp_multiclass_manual_prediction(testPoint,training_data,params_for_gp_toolbox,trainedSystem,KWii);
       min_mu = min(min_mu,meanvec);
       max_mu = max(max_mu,meanvec);
       min_var = min(min_var,covmat);
       max_var = max(max_var,covmat);
   end
end

end