% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% Setup environment
%  Add common directory to your path for minFunc
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));
addpath(genpath('.'));
addpath(genpath('../preprocessing'));

% Run preprocess code if necessary
if ~exist('nimstim_data', 'var')
    prepare_data;
end    
    

%% Load the NimStim Data
data = nimstim_data;
labels = nimstim_labels;

i = randperm(length(labels));
data   = data(:,i);
labels = labels(i); 

% Partition data
train_X = data(:,1:255);
train_y = labels(1:255);

test_X = data(:,256:341);
test_y = labels(256:341);

% Training set info
m = size(train_X,2);
n = size(train_X,1);

K = length(unique(labels));


%% Load the POFA Data
% data = pofa_data;
% labels = pofa_labels;
% 
% i = randperm(length(labels));
% data   = data(:,i);
% labels = labels(i); 
% 
% % Partition data
% train_X = data(:,1:72);
% train_y = labels(1:72);
% 
% test_X = data(:,73:96);
% test_y = labels(73:96);
% 
% % Training set info
% m = size(train_X,2);
% n = size(train_X,1);
% 
% K = length(unique(labels));


%% Network hyperparameters
% ei is a structure you can use to store hyperparameters of the network
ei = [];
ei.input_dim = n;
ei.output_dim = K;
ei.layer_sizes = [30, 20, ei.output_dim];
ei.lambda = 1e-6;
ei.activation_fun = 'logistic';
%ei.activation_fun = 'tanh';


%% Setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% Setup minFunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';
options.maxIter = 1000;

%% Training with minFunc
b_min = cputime;
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, train_X, train_y);
e_min = cputime - b_min;
disp(e_min);

%% Training with stochastic gradient descent
alpha = 0.01;
b_sgd = cputime;
[opt_params, error] = stochastic_grad_desc(@supervised_dnn_cost, params, alpha, 100, train_X, train_y, test_X, test_y, ei); 
e_sgd = cputime - b_sgd;
disp(e_sgd);

%% Accuracy on test and train set.
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, test_X, [], true);
[~,pred] = max(pred);
acc_test = mean(pred==test_y);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, train_X, [], true);
[~,pred] = max(pred);
acc_train = mean(pred==train_y);
fprintf('train accuracy: %f\n', acc_train);





