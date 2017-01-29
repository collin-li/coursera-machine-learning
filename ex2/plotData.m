function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create new figure
figure; hold on;

% Find indices of positive and negative examples
pos = find(y == 1);
neg = find(y == 0);

% Plot positives with big ('MarkerSize', 7), bold ('LineWidth', 2), black pluses ('k+')
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);

% Plot negatives with large ('MarkerSize', 7), circles ('ko') with yellow fill ('MarkerFaceColor', 'y')
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

hold off;

end
