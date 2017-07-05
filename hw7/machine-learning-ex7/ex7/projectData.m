function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

% recall the PCA alg. flow
% 1. Preprocessing, 2. Calculate sigma (covariance matrix) 
% 3. Calculate eigenvectors with svd, 4. Take k vectors from U (Ureduce= U(:,1:k);) 
% 5. Calculate z (z = Ureduce' * x;)

% for each sample
for i = 1:size(X, 1)
    sample_vect_x = X(i, :)';
    
    % translate the features into the new coordinate sys. defined by the k eigenvectors
    for j = 1:K
        eigen_vect_j = U(:, j);
        
        projection_k_value = sample_vect_x' * eigen_vect_j;
        
        Z(i, j) = projection_k_value;
    end
end



% =============================================================

end
