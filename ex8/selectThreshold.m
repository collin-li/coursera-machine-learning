function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % Compute predictions
    predictions = pval < epsilon;
    
    % Compute F1 score
    tp = sum( (yval == 1) & (predictions == 1) ); % Number of true positives
    fp = sum( (yval == 0) & (predictions == 1) ); % Number of false positives
    fn = sum( (yval == 1) & (predictions == 0) ); % Number of false negatives
    prec = tp / (tp + fp); % Precision metric
    rec = tp / (tp + fn); % Recall metric
    F1 = 2*prec*rec / (prec + rec);

    % Check for improved F1 score
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
