function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%


% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% init first layer w/ bias terms for each sample
a1 = [ones(m, 1) X];

% each sample feature row forms wieghted eq. with Ø1's params.
z2 = a1 * Theta1';
a2 = sigmoid(z2);
% add bias terms to each of the a2 sample outputs
a2 = [ones(size(a2,1), 1) a2];

% each sample feature row forms wieghted eq. with Ø2's params.
z3 = a2 * Theta2';
a3 = sigmoid(z3);
% final hypothesis vector
hThetaX = a3;

% init response vector
yVec = zeros(m, num_labels);
% set ith index of yVec to 1
for i = 1:m
    yVec(i, y(i)) = 1;
end

% iterative version of cost function calculation
% for i = 1:m
%     
%     term1 = -yVec(i,:) .* log(hThetaX(i,:));
%     term2 = (ones(1,num_labels) - yVec(i,:)) .* log(ones(1,num_labels) - hThetaX(i,:));
%     k_output_sample_summ = sum(term1 - term2);
%     J = J + k_output_sample_summ;
%     
% end
% J = J / m;

% calculating cost function
tot_output_sample_summ = sum(-yVec .* log(hThetaX) - (1 - yVec) .* log(1-hThetaX));
tot_samples_summ = sum(tot_output_sample_summ);
J = 1/m * tot_samples_summ;





% Part 2: Implement the backpropagation algorithm _to compute the gradients_
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

for t = 1:m

	% For the input layer, where l=1:
	a1 = [1; X(t,:)'];

	% For the hidden layers, where l=2:
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];

	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	yy = ([1:num_labels]==y(t))';
  
	% For the err delta values:
	delta_3 = a3 - yy;

	delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)];
	delta_2 = delta_2(2:end); % Taking off the bias row

	% note: delta_1 is not calculated because we do not associate error with the input    

	% BigDelta update
  big_delta_1 = Theta1_grad + delta_2 * a1';
  big_delta_2 = Theta2_grad + delta_3 * a2';
  
  % now have the gradients for each ø on J(Ø)
	Theta1_grad = big_delta_1;
	Theta2_grad = big_delta_2;
end




% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

regularator = (lambda/(2*m)) * ...
              (sum(sum(Theta1(:,2:end) .^ 2)) ...   % summ over all features for each sample
                + sum(sum(Theta2(:,2:end) .^ 2)));  % summ over all features for each 'hidden sample'
J = J + regularator;

Theta1_grad = (1/m) * big_delta_1 + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * big_delta_2 + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
