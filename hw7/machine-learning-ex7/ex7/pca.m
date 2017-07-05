function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

% recall the PCA alg. flow
% 1. Preprocessing, 2. Calculate sigma (covariance matrix) 
% 3. Calculate eigenvectors with svd, 4. Take k vectors from U (Ureduce= U(:,1:k);) 
% 5. Calculate z (z =Ureduce' * x;)

cov_mat = 1/m * (X' * X);
[U, S, V] = svd(cov_mat);




% =========================================================================

end
