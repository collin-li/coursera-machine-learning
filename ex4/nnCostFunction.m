function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         

%% =========== Part 1: Feedforward the neural network to compute cost =============

% Feedforward input layer (1) to hidden layer (2)
X = [ones(m, 1) X]; % Add bias parameter
a2 = sigmoid(X * Theta1'); % Returns m x n2 matrix (n2 = # of features in hidden layer)

% Feedforward hidden layer (2) to output layer (3)
a2 = [ones(m, 1) a2]; % Add bias parameter
h = sigmoid(a2 * Theta2'); % Returns m x k matrix (k = # of classes in output layer)

% Compute cost function
J = 0; % Initialize cost counter
for k = 1:num_labels % Loop through each classifier
    yk = y == k; % m x 1 boolean vector of observations by class
    hk = h(:,k); % m x 1 vector of predictions by class
    
    J = J + 1/m * (-yk'*log(hk) - (1-yk)'*log(1-hk)); % Cost function by class


%% =========== Part 2: Backpropagation algorithm to compute gradients =============

    delta3(:,k) = hk - yk;  % Compute error term of output layer
end

% Backpropagate from output layer to hidden layer
z2 = [ones(m,1) X*Theta1']; % Add bias parameter
delta2 = delta3*Theta2 .* sigmoidGradient(z2); % Backpropagation
delta2 = delta2(:,2:end); % Remove bias parameters

% Note: delta1 not required since X is input layer

% Compute intermediate gradient term
Delta1 = delta2'*X; % Returns n2 x n1 matrix
Delta2 = delta3'*a2; % Returns k x n2 matrix

% Partial derivatives of cost function with respect to Theta1 and Theta2 respectively
Theta1_grad = 1/m * Delta1;
Theta2_grad = 1/m * Delta2;
% Note: Can check against numerical method by running checkNNGradients(0)


%% =========== Part 3: Regularization of cost function and gradients =============

% Exclude bias parameters from regularization
Theta1_reg = Theta1; Theta2_reg = Theta2;
Theta1_reg(:,1) = 0;
Theta2_reg(:,1) = 0;

% Add cost to regularize NN features
Theta_reg = [Theta1_reg(:) ; Theta2_reg(:)]; % Unroll features
J = J + lambda/(2*m) * Theta_reg'*Theta_reg; % Add feature costs

% Add impact of regularization to cost gradients
Theta1_grad = Theta1_grad + lambda/m*Theta1_reg;
Theta2_grad = Theta2_grad + lambda/m*Theta2_reg;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
