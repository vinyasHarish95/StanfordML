function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

h = sigmoid(X*theta); % Creates vector of hypotheses

j = zeros(size(y));

% Creates vector of costs for each training example 
j = (((-y).*log(h))-((1-y).*log(1-h))); 

% Folds cost vector to compute total cost w regularization, but does not 
% regularize the first theta (0 in math, 1 in MATLAB indexing) 
J = (sum(j)/m) +((lambda/(2*m))*sum(theta(2:end).^2)); 

% Compute gradient ensuring to not regularize the first theta
grad = ((1/m) * sum((h - y) .* X))+((lambda/m).*(theta'));
grad(1) = ((1/m) * sum((h - y) * X(1)));

end
