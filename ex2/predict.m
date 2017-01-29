function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

h = sigmoid(X*theta); % Logistic model predictions (m x 1)
p = h >= 0.5; % Binary prediction based on a threshold at 0.5

end
