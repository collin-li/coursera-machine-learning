function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Add ones to the X data matrix
X = [ones(m, 1), X];

% Logistic model predictions for the hidden layer
a2 = sigmoid(X*Theta1');
a2 = [ones(m, 1), a2]; % Add bias parameter

% Find index (corresponding to each class) of maximum prediction for each row (example)
[~, p] = max((a2*Theta2'), [], 2);

end
