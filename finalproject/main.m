ei = [];
ei.input_dim = 784;
ei.layer_sizes = [500, 200, 2];

filename = './MNIST/train-images.idx3-ubyte' ;
fulldata = loadMNISTImages(filename);
fulldata = fulldata'; 
numtrain = 50000;
numtest = 5000;
train = fulldata(1:numtrain,:);
test = fulldata(numtrain+1:numtrain + numtest,:);

stack_rbm_2d = rbm_initialize(ei, train);
stack_sgd_2d  = sgd_train(train, unfold_stack(stack_rbm_2d));
save stack_rbm_2d stack_sgd_2d;

ei = [];
ei.input_dim = 784;
ei.layer_sizes = [500, 200, 30];
stack_rbm = rbm_initialize(ei, train);
stack_sgd  = sgd_train(train, unfold_stack(stack_rbm));
save stack_rbm stack_sgd;