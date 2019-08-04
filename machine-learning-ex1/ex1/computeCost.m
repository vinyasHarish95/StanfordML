function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

J = 0;

h = X * theta; % Creates vector of hypotheses
j = ((h-y).^2); % Creates vector of costs for each training example
J = sum(j)/(2*m); % Folds cost vector to compute total cost 

end
