function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

U_reduce = U(:, 1:K); % Select top K eigenvectors (size n x K)
Z = X*U_reduce; % Return projections (size m x K)

end
