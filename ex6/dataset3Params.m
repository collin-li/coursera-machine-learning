function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Test values of C and sigma
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';

% Loop through each combination of C and sigma
for c = 1:length(C_vec)

    C = C_vec(c); % Set value of C
    
    for s = 1:length(sigma_vec)
        sigma = sigma_vec(s); % Set value of sigma
        
        % Train SVM with Gaussian kernel
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        
        % Return predictions on cross validation set
        predictions = svmPredict(model, Xval);

        % Compute and store prediction error on cross validation set
        error_val(c,s) = mean(double(predictions ~= yval));
    end
    
end

% Identify lowest error combination of C and sigma
[c, s] = find(min(error_val(:)) == error_val);
C = C_vec(c);
sigma = sigma_vec(s);

end
