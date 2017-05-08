function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Vectorizing the Cost function
h = sigmoid(X * theta);             % hypothesis function
A = -y' * log(h);                   % first term in Cost function
B = (1-y)' * log(1-h);              % second term in Cost
unreg = (1/m) * (A - B);            % unregularized cost

theta(1) = 0;                               % exclude the bias variable
reg = (lambda/(2*m)) * (theta' * theta);    % regularization term
 
J = unreg + reg;

% Vectorizing the gradient calculation
grad = (1/m) * (h - y)' * X;                % take the gradient
grad = grad' + (lambda/m)*theta;            % add regularized gradient term




% =============================================================

end
