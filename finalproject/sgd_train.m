function [ stack_out ] = sgd_train( traindata , stack )
totalsamples = size(traindata,1);
minibatchsize = 1000;
numepochs = 200;
numSGDsteps = 50;
epsilon = 0.1;
momentum = 0.5;
lambda = 0.0002;
numbatches = floor(totalsamples/minibatchsize);
inputdim = size(traindata,2);
%scramble the order of the training data to create minibatches
traindata_scrambled = traindata(randperm(totalsamples),:);
batches = ones(numbatches, minibatchsize, inputdim);
%create minibatches
for i = 1:numbatches
    batches(i,:,:) = traindata_scrambled((i-1)*minibatchsize + 1:i*minibatchsize,:);
end

numlayers = length(stack) + 1;
delta = cell(1, numlayers);
for i = 1:numlayers-1
    delta{i}.dw = zeros(size(stack{i}.W));
    delta{i}.db = zeros(size(stack{i}.b));
end
epocherrors_sgd = [];
%iterate over minibatches
for epoch = 1:numepochs
    epoch_error = 0;
    if epoch > 0.6*numepochs
        momentum = 0.5;
    end
for i=1:numSGDsteps
    %pick random minibatch
    batchnumber = randi([1 numbatches],1);
    batchdata = reshape(batches(batchnumber,:,:),[minibatchsize inputdim]);
    [recon_error,stack, delta] = backpropagation(batchdata, stack, delta, epsilon, momentum, lambda);
    epoch_error = epoch_error + recon_error;
end
epocherrors_sgd = [epocherrors_sgd epoch_error];
fprintf(1, 'epoch %d error %6.4f  \n', epoch, epoch_error); 
end
stack_out = stack;
%save epocherrors_sgd
end