function [ stack ] = initialize_weights( ei, inputdata )
%INITIALIZE_WEIGHTS Random weight structures for a network architecture
%   eI describes a network via the fields layerSizes, inputDim, and outputDim 
%   
%   This uses Xavier's weight initialization tricks for better backprop
%   See: X. Glorot, Y. Bengio. Understanding the difficulty of training 
%        deep feedforward neural networks. AISTATS 2010.

%% initialize hidden layers
stack = cell(1, numel(ei.layer_sizes));

%1use train data
%2make minibatches
%3consider only two layers
%4initialize the weights and biases to start training the RBM
%5use each minibatch to compute a forward pass that computes an average over
%the positive gradient. For this use real values of inputs and
%probabilities of hidden states, not the binary states
%6now for each vector in the minibatch construct the negative gradient by
%using the real valued visible input to generate stochastic binary hidden
%states. The binary hidden states are used to generate real valued
%reconstructions and the process is repeated. In the last step, we use
%probabilities for hidden layer to construct the negative gradient.
%and then average the negative gradient over the minibatch vectors.
%7update the weights and biases for every minibatch
%8generate new train data by passing entire training data on the trained
%RBM and repeat the process

currdata = inputdata;
numsamples = size(inputdata,1);
for l = 1 : numel(ei.layer_sizes)    
    l
    if l ==  numel(ei.layer_sizes)
        S = rbm_twolayer_linear(currdata, ei.layer_sizes(l));
    else
        S = rbm_twolayer(currdata, ei.layer_sizes(l));
    end
    stack{l}.W = S.W;
    stack{l}.a = S.a;
    stack{l}.b = S.b;   
    currdata = sigmoid(repmat(S.b,numsamples,1) + currdata*S.W);
end