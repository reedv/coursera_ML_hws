function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_vect = X * theta;
err_vect = h_vect - y;
samples_summ = sum(err_vect .^ 2);
J_unreg = 1/(2*m) * samples_summ;

J_reg_term = lambda / (2*m) * sum(theta(2:end) .^ 2);

J = J_unreg + J_reg_term;



grad_reg_term = (lambda/m) .* theta;
grad_reg_term(1) = 0; % this is always 0
X_feature_rows = X';  % row i holds all values of ith feature for each sample in X

grad = ((1/m) .* (X_feature_rows * err_vect)) + grad_reg_term;










% =========================================================================

grad = grad(:);

end
