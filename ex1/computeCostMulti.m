function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% Compute the cost of a particular choice of theta
h = X*theta; % Linear model predictions (m x 1)
J = 1/(2*m) * (h-y)'*(h-y); % Vectorized cost function

end
