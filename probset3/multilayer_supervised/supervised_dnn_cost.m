function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%   Does all the work of cost / gradient computation

%% Determine activation function type.
if strcmp(ei.activation_fun, 'logistic')
    f            = @sigmoid_activation;
    f_derivative = @(A) (A.*(1-A));
elseif strcmp(ei.activation_fun, 'tanh')
    f            = @tanh_activation;
    f_derivative = @(A) ( 1-A.^2); 
end


%% Default values.
po = false;
if exist('pred_only','var')
  po = pred_only;
end;


%% Reshape into network.
stack     = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct      = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);


%% Forward propagation.
% Initial activation is simply the data input
hAct{1}.activation = data;

% Using prior layers, compute activations
for h = 1:numHidden
    Z = bsxfun(@plus, stack{h}.W * hAct{h}.activation, stack{h}.b);
    Z = f(Z);
    hAct{h+1}.activation = Z;
end

% Output layer
output_layer = numHidden + 1;
Z_output = bsxfun(@plus, stack{output_layer}.W * hAct{output_layer}.activation, stack{output_layer}.b);
pred_prob = softmax(Z_output);


%% Return here if only predictions (po == True) desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;


%% Compute cost.
[~, m] = size(data);

cost = 0;
for i = 1:m
    cost = cost - log(pred_prob(labels(i), i));
end
cost = cost/m;

%% Compute gradients using backpropagation algorithm.
deltas       = cell(output_layer, 1);

% Compute the delta matrix for output layer
I = eye(size(pred_prob, 1));
deltas{output_layer}.delta_matrix = pred_prob - I(:, labels); 

% Compute the delta matrices for hidden layers, h hidden layers
for h = numHidden: -1 : 1
    deltas{h}.delta_matrix =  (stack{h + 1}.W' * deltas{h + 1}.delta_matrix) .* f_derivative(hAct{h + 1}.activation);
end

% Compute the gradients
for h = 1:(output_layer)
    % Gradients for weight_matrix
    gradStack{h}.W = deltas{h}.delta_matrix * hAct{h}.activation';
    
    % Gradients for bias
    gradStack{h}.b = sum(deltas{h}.delta_matrix, 2);
    
end

% Regularization impact to Cost Function and Weight/Bias Updates
for h = 1:(output_layer)
    % Modify cost function to include Weight Decay Term
    cost = cost + 1/2 * ei.lambda * sum(sum(stack{h}.W.^2));
    
    % Change the W and b updates according to regularization
    gradStack{h}.W = 1/m * gradStack{h}.W + ei.lambda * stack{h}.W;
    gradStack{h}.b = 1/m * gradStack{h}.b;
end


%% Reshape gradients into vector.
[grad] = stack2params(gradStack);
end



