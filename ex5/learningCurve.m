function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda, random)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%       If 'random' is non-zero, errors will be based on random samples 
%       instead of the first i values in the data-set (default is 0)
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% Check for randomization requirement
if ~exist('random', 'var')
    random = 0; % Default value
end

if random == 0 % =============== Draw non-random samples ===============
    % Loop through each training set size
    for i = 1:m
        % Define training sets
        X_train = X(1:i, :);
        y_train = y(1:i);

        % Train linear regression for each training set size
        theta = trainLinearReg(X_train, y_train, lambda);

        % Store training and cross validation errors in a m x 1 vector
        [error_train(i), ~] = linearRegCostFunction(X_train, y_train, theta, 0);
        [error_val(i), ~] = linearRegCostFunction(Xval, yval, theta, 0);
    end
    
else % =============== Draw random samples ===============
    % Number of validation examples
    m_val = size(Xval, 1);

    % Number of random samples to draw
    num_iter = 50;

    for n = 1:num_iter
        for i = 1:m
            % Define training set
            index_train = randperm(m)';
            index_train = index_train(1:i);
            X_train = X(index_train,:);
            y_train = y(index_train);

            % Train linear regression for each training set size
            theta = trainLinearReg(X_train, y_train, lambda);

            % Define validation set
            index_val = randperm(m_val)';
            index_val = index_val(1:min(i,m_val));
            Xval_iter = Xval(index_val,:);
            yval_iter = yval(index_val);

            % Store training and cross validation errors in a m x n vector
            [error_train(i,n), ~] = linearRegCostFunction(X_train, y_train, theta, 0);
            [error_val(i,n), ~] = linearRegCostFunction(Xval_iter, yval_iter, theta, 0);
        end
    end

    % Take average error across random samples
    error_train = mean(error_train, 2);
    error_val = mean(error_val, 2);
end

end
