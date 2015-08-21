function [act] = activate(samples, stack)
numlayers = length(stack)+1;
numsamples = size(samples,1);
act = cell(1, numlayers);
act{1}.z = samples;
act{1}.a = act{1}.z;
for i = 2:numlayers
    act{i}.z =  repmat(stack{i-1}.b,numsamples,1) + act{i-1}.a*stack{i-1}.W;
    if i~=4
        act{i}.a = sigmoid(act{i}.z);
    else
        act{i}.a = act{i}.z;
    end
end
    
end