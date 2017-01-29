function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Set initial theta
initial_theta = zeros(n + 1, 1);

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Loop through to train theta values for each classifier
for c = 1:num_labels
    % Run fmincg to obtain the optimal theta (returns theta and cost)
    [all_theta(c,:), cost] = ...
        fmincg(@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options); 
        % Use (y == c) to obtain a boolean vector for each class
end

end
