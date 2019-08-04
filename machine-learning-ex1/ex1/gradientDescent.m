function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    theta_0 = theta(1);
    theta_1 = theta(2);

    h = X * theta; % Compute hypothesis vec
    
    % Compute feature weights
    theta_0 = theta_0 - alpha * (1/m) * sum(h - y);   
    theta_1 = theta_1 - alpha * (1/m) * sum((h - y) .* X(:,2));
    
    % Update feature vec with new weights 
    theta = [theta_0 ; theta_1];
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
end

end
