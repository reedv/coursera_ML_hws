function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


for i = 1:size(centroids, 1)
    % vecotr encoding which samples group w/ centroid i
    c_i = idx==i;
    % get total number of samples grouped w/ centroid i 
    n_i = sum(c_i);
    
    % get row of n 1s for rows grouped w/ centroid i, else row of n 0s
    c_i_matrix = repmat(c_i, 1,n);
    % multiply row-wise to get only samples grouped w/ centroid i, else 0s
    X_c_i = X .* c_i_matrix;
    
    summed_feaures_row = sum(X_c_i);
    
    % update values of ith centroid corresponding to each sample dimension/feature
    centroids(i,:) = summed_feaures_row ./ n_i;
end




% =============================================================


end

