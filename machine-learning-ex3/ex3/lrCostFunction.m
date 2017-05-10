function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% Identifcal function to costFunctionReg() from ex2
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

grad = grad(:);

end
