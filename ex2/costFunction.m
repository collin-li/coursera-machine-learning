function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% Compute the cost for a particular choice of theta
h = sigmoid(X*theta); % Logistic model predictions (m x 1)
J = 1/m * (-y'*log(h) - (1-y)'*log(1-h)); % Vectorized cost function

% Compute gradient of cost for a particular choice of theta
grad = 1/m * X'*(h-y); % Vectorized gradient of the cost function

end
