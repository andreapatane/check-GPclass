function Y = inv_chol(R)
% Matrix Inversion using Cholesky Decomposition
%
% Finds the inverse of the matrix X, given its (lower triangular) Cholesky
% Decomposition; i.e. X = LL', according to the paper 'Matrix Inversion
% Using Cholesky Decomposition', Aravindh Krishnamoorthy, Deepak Menon,
% arXiv:1111.4144.
%

% Version 0.1, 2013-05-25, Aravindh Krishnamoorthy
% e-mail: aravindh.k@ieee.org

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initializations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = size(R, 1) ;
%Y = zeros(N, N) ;
% Construct the auxillary diagonal matrix S = 1/rii
Y = inv(diag(diag(R))) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=N:-1:1
    for i=j:-1:1
        Y(i,j) = Y(i,j) - R(i,i+1:end)*Y(i+1:end,j) ;
        Y(i,j) = Y(i,j)/R(i,i) ;
        % Write out the symmetric element
        Y(j,i) = Y(i,j);
    end
end

end