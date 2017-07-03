function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    % øj := øj - a*1/m*∑( (hø(xi) - yi) * xj ), where hø(X) = X * theta
    % j = jth feature, i = ith sample
    
    % vecotr of predictions for each sample
    H = X * theta;
    % vector of prediction errors for each sample
    samples_err_vect = H - y;
    % multiply all features in ith sample by prediction err. for that sample
    features_weighted_mat = samples_err_vect .* X;
    % sum cols. of weighted features samples and convert to single col. vector
	% (by default, sum() summs by col.)
    delta_row = (sum(features_weighted_mat));
	delta_vect = delta_row';
        
    theta = theta - (alpha/m * delta_vect);
        


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
