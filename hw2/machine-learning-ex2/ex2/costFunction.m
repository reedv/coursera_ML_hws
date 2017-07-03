function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

H = X * theta;

response_1_cost_vect = -y .* log(sigmoid(H));
response_0_cost_vect = (1 - y) .* log(1 - sigmoid(H));
sample_costs_vect = response_1_cost_vect - response_0_cost_vect; 

J = 1/m * sum(sample_costs_vect);


sample_errs_vect = sigmoid(H) - y;
samples_features_weighted = sample_errs_vect .* X;
summed_feature_weights = sum(samples_features_weighted);

grad = 1/m .* summed_feature_weights;







% =============================================================

end
