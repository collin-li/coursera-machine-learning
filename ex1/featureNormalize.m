function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = mean(X); % Average of each feature (1 x n)
sigma = std(X); % Standard deviation of each feature (1 x n)
X_norm = (X - mu) ./ sigma % Normalized version of X (m x n)

end
