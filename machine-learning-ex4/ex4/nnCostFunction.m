function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Step 1 - Expand the 'y' output values into a matrix of single values
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

% Step 2 - Forward Propogation (page 6 in slides)
a1 = [ones(m,1), X];
z2 = a1 * Theta1';
a2 = [ones(m,1), sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3); % h_theta = a3

% Step 3 - Cost Function, non-regularized
% Hint: Terms which lie on the main diagonal are the same terms that result 
% from the double-sum of the element-wise product.
% Compute the sum of the diagonal elements using the "trace()" 
% command, or by sum(sum(...)) after element-wise multiplying by an 
% identity matrix of size (K x K). 

unreg = (1/m) * trace((-y_matrix' * log(a3) - (1-y_matrix)' * (log(1-a3))));

% Alternatively
% J2 = (1/m) * sum(sum((-y_matrix .* log(a3) ...
%            - (1-y_matrix) .* (log(1-a3)))));

% Step 4 - Cost Regularization (using formula on page 6)
% Important - ignore columns of bias units!
reg = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))); 

J = unreg + reg; % add regularized cost to unregularized cost
    
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%


% look at tutorial for ex4
% 1. forward propagation already done
d3 = a3 - y_matrix; % 2.
z2 = a1 * Theta1'; % 3.

% 4. no bias, so start from 2nd column
% d2 is the product of d3 and Theta2(no bias), then element-wise scaled by 
% sigmoid gradient of z2.
d2 = sigmoidGradient(z2) .* (d3 * Theta2(:,2:end));

Delta1 = d2' * a1; % 5.
Delta2 = d3' * a2; % 6.

% 7. Scale by 1/m
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

% 8. set the first column of Theta1 and Theta2 to all-zeros
Theta1(:,1) = 0;
Theta2(:,1) = 0;

% 9. Scale each Theta matrix by ?/m
Theta1 = (lambda/m) * Theta1;
Theta2 = (lambda/m) * Theta2;

% 10. Add each of these modified-and-scaled Theta matrices to 
% the un-regularized Theta gradients that you computed earlier.
Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
