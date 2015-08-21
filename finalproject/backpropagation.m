function [ f,  stack, delta ] = backpropagation(samples, stack, delta_prev, epsilon, momentum, lambda)
recon =  activate(samples, stack);
numsamples = size(samples,1);
numlayers= length(stack)+1;
outputs = recon{numlayers}.a;
f = -(1.0/numsamples)*sum(sum(samples.*log(outputs) + (1-samples).*log(1-outputs)));
delta = cell(1, numlayers);
delta{numlayers}.del = (1/numsamples)*(outputs - samples);
for i = numlayers-1:-1:1
    if i ~= 4 
        delta{i}.del = (delta{i+1}.del*stack{i}.W').*(sigmoid(recon{i}.z).*(1 - sigmoid(recon{i}.z)));
    else
        delta{i}.del = (delta{i+1}.del*stack{i}.W');  
    end  
    delta{i}.dw = -epsilon*(recon{i}.a'*delta{i+1}.del + lambda*stack{i}.W) + momentum*delta_prev{i}.dw;
    delta{i}.db = -epsilon*(ones(1,numsamples)*delta{i+1}.del)+ momentum*delta_prev{i}.db ; 
end

for i = numlayers-1:-1:1
    stack{i}.W = stack{i}.W + delta{i}.dw;
    stack{i}.b = stack{i}.b + delta{i}.db;
end

end

