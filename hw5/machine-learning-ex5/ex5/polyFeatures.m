function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

num_samples = size(X,1);
for i=1:num_samples
  poly_ith_feature_vect = zeros(p,1);
  
  for j=1:p
    poly_ith_feature_vect(j) = X(i) .^ j;
  end

  % ith row in X_poly holds ith feature of X up to deg. p
  X_poly(i,:) = poly_ith_feature_vect';


% =========================================================================

end
