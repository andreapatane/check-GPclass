function out = softmax_lik(logit_vals,classIdx)

logit_vals = logit_vals - max(logit_vals);
out = exp(logit_vals);

out = out(classIdx)/sum(out);


end