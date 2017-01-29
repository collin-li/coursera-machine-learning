function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population (x) and profit (y).

figure; % Open a new figure window

plot(x, y, 'rx', 'MarkerSize', 10); % Plots training data with large ('MarkerSize', 10), red crosses ('rx')
ylabel('Profit in $10,000s'); % Set y-axis label
xlabel('Population of City in 10,000s'); % Set x-axis label

end
