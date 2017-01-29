function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

h = sigmoid(X*theta); % Logistic model predictions (m x 1)
theta_reg = [0; theta(2:end)]; % Exclude bias parameter from regularization

% Compute the cost, with regularization, for a particular choice of theta
J = 1/m * (-y'*log(h) - (1-y)'*log(1-h)) + lambda/(2*m) * theta_reg'*theta_reg;

% Compute gradient of cost for a particular choice of theta
grad = 1/m * X'*(h-y) + lambda/m * theta_reg;

end
