function [ stack ] = rbm_twolayer_linear( traindata , hiddendim )
%rbm_two layer RBM training for two layers given the training set
stack = [];
totalsamples = size(traindata,1);
minibatchsize = 100;
numepochs = 100;
numSGDsteps = 500;
cdsteps = 1;
epsilon = 0.001;
momentum = 0.5;
lambda = 0.00002;
numbatches = floor(totalsamples/minibatchsize);
visibledim = size(traindata,2);
%scramble the order of the training data to create minibatches
traindata_scrambled = traindata(randperm(totalsamples),:);
batches = ones(numbatches, minibatchsize, visibledim);
%create minibatches
for i = 1:numbatches
    batches(i,:,:) = traindata_scrambled((i-1)*minibatchsize + 1:i*minibatchsize,:);
end
%initialize weights
W =  0.1*randn(visibledim, hiddendim);
pixprob = mean(traindata,1);
a =  log((pixprob + 0.01)./(1 - pixprob));
a =  zeros(1,visibledim);
b = zeros(1,hiddendim);

%iterate over minibatches
errW = 0;
erra = 0;
errb = 0;

for epoch = 1:numepochs
    epoch_error = 0;
    if epoch > 0.6*numepochs
        momentum = 0.5;
    end
for i=1:numSGDsteps
    %fprintf(1,'SGD step number %d\r',i);
    %pick random minibatch
    batchnumber = randi([1 numbatches],1);
    
    %compute positive gradient    
    visbatch = reshape(batches(batchnumber,:,:),[minibatchsize visibledim]);
    hiddenbatch = repmat(b,minibatchsize,1) + visbatch*W;
    posstatW = (visbatch'*hiddenbatch)/minibatchsize;
    posstata = mean(visbatch,1);
    posstatb = mean(hiddenbatch,1);
    
    visbatch_data = visbatch;
    %compute negative gradient
    hiddenbatch = hiddenbatch + randn(minibatchsize,hiddendim);
    for j = 1:cdsteps        
        visbatch = sigmoid(repmat(a,minibatchsize,1) + hiddenbatch*W');
        hiddenbatch = repmat(b,minibatchsize,1) + visbatch*W;
    end
    recon_error_xent =  -(1.0/minibatchsize)*sum(sum(visbatch_data.*log(visbatch) + (1-visbatch_data).*log(1-visbatch)));
    %recon_error_sos = sum(sum((visbatch - visbatch_data).^2))/(minibatchsize*visibledim);
    epoch_error = epoch_error + recon_error_xent;
    
    negstatW = (visbatch'*hiddenbatch)/minibatchsize;
    negstata = mean(visbatch,1);
    negstatb = mean(hiddenbatch,1);
    
    %update weights and biases
    errW = epsilon*(posstatW - negstatW - lambda*W) + momentum*errW;
    erra = epsilon*(posstata - negstata - lambda*a) + momentum*erra;
    errb = epsilon*(posstatb - negstatb - lambda*b) + momentum*errb;
    
    W = W + errW;
    a = a + erra;
    b = b + errb;    
end
fprintf(1, 'epoch %d error %6.4f  \n', epoch, epoch_error); 
end
stack.W =W;
stack.a =a;
stack.b =b;
end

