function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

errorRow = 0;

potential_Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30,];
potential_sigmas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
results = zeros(size(potential_Cs,2)*size(potential_sigmas,2), 3);

for curr_C = potential_Cs
    for curr_sigma = potential_sigmas;
    
        errorRow = errorRow + 1;
        
        % train a SVM using current C and sigma values
        curr_model = svmTrain( ...
          X, y, curr_C, ...
          @(x1, x2) gaussianKernel(x1, x2, curr_sigma) ...
        );
        
        % get validation results from current model
        predictions = svmPredict(curr_model, Xval);
        
        % get total mean error of all predictions
        prediction_errs_count = predictions != yval;
        prediction_error = mean(double(prediction_errs_count));

        % store C and sigma info realted to this error
        results(errorRow, :) = [curr_C, curr_sigma, prediction_error]; 
        
    end
end

% sort matrix by column #3, the error, ascending
sorted_results = sortrows(results, 3); 

% get optimal param. values
C = sorted_results(1, 1);
sigma = sorted_results(1, 2);




% =========================================================================

end
