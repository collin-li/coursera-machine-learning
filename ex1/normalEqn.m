function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

% Matrix size of theta is n x 1 (n = number of features)
%   X is m x n (m = number of examples)
%   y is m x 1
theta = pinv(X' * X) * X' * y; % Normal equation for linear regression

end
