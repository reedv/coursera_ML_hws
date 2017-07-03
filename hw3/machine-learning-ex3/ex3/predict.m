function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% feedforward propagation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%

% append bias terms to each input sample
a1_input_rows = [ones(m, 1) X];

%size(a1)
%size(Theta1)

% convert Theta1 to matrix of theta columns for each activation unit
Theta1_vects = Theta1';
z2 = a1_input_rows * Theta1_vects;
a2_units_rows = sigmoid(z2);

%size(a2)

% append bias terms to 
a2_units_rows = [ones(size(a2,1), 1) a2_units_rows];

% convert Theta2 to matrix of theta columns for each activation unit
Theta2_vects = Theta2';
z3 = a2_units_rows * Theta2_vects;
a3_units_rows = sigmoid(z3);

%size(a3)

% get label corresponding to highest value hypoth. for each row of hypotheses
[val, index] = max(a3_units_rows, [], 2);

p = index;







% =========================================================================


end
