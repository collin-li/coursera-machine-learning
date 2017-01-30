function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms

% Randomly initialize the weights to small values
epsilon_init = round( (6 / (L_in + L_out))^0.5 , 2);
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

end
