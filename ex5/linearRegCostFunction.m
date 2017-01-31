function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% Compute the cost of a particular choice of theta
h = X*theta; % Linear model predictions (m x 1)
J = 1/(2*m) * (h-y)'*(h-y); % Vectorized cost function

% Add cost to regularize features
theta_reg = [0; theta(2:end)]; % exclude bias parameter from regularization
J = J + lambda/(2*m) * theta_reg'*theta_reg; % Vectorized regularization cost

% Compute gradient of cost for a particular choice of theta
grad = 1/m * X'*(h-y); % Vectorized gradient of the cost function

% Add impact of regularization to gradient of cost
grad = grad + lambda/m * theta_reg;

grad = grad(:); % Unroll gradients

end
