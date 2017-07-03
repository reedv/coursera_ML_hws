function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

H = X * theta;

% cal. J unregularized
response_1_cost_vect = -y .* log(sigmoid(H));
response_0_cost_vect = (1 - y) .* log(1 - sigmoid(H));
sample_costs_vect = response_1_cost_vect - response_0_cost_vect; 

J_unreg = 1/m .* sum(sample_costs_vect);

% add regularizing J term
J_reg_term = lambda/(2*m) .* sum( theta(2:size(theta,1), :) .^ 2 );

J = J_unreg + J_reg_term;


% calc. grad unregularized
sample_errs_vect = sigmoid(H) - y;
samples_features_weighted = sample_errs_vect .* X;
summed_feature_weights = sum(samples_features_weighted);

grad_unreg_rowVect = 1/m .* summed_feature_weights;

% add grad regularizing term
grad = grad_unreg_rowVect;
grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m)*theta(2:length(theta))';

% =============================================================

end
