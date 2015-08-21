function [ out ] = reconstruct( samples, stack )
numlayers = length(stack);
samplesize = size(samples,1);
for l=1:length(stack)
W = stack{l}.W;
a = stack{l}.a;
b = stack{l}.b;
if l == length(stack)
    samples = repmat(b,samplesize,1) + samples*W;
else
    samples = sigmoid(repmat(b,samplesize,1) + samples*W);
end

end
for l=length(stack):-1:1
W = stack{l}.W;
a = stack{l}.a;
b = stack{l}.b;
samples = sigmoid(repmat(a,samplesize,1) + samples*W');
end
out = samples;
end

