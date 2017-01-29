function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Import cost function and gradient without regularization
[J, grad] = costFunction(theta, X, y);

% Add cost to regularize features
theta_reg = [0; theta(2:end)]; % exclude bias parameter from regularization
J = J + lambda/(2*m) * theta_reg'*theta_reg; % Vectorized regularization cost

% Add impact of regularization to gradient of cost
grad = grad + lambda/m * theta_reg;

end
