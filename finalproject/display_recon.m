output = [];
samples = size(test,1);
randperm(samples)
[coeff,score,latent] = pca(train);
test_decom = test*coeff(:,1:30);
test_recon_pca = test_decom*(coeff(:,1:30)');
p = activate(test,unfold_stack(stack_rbm));
q = activate(test,stack_sgd);
testerror_sos = sum(sum((test - reconstruct(test, stack_rbm)).^2))/(numtest);
testerror_sos2 = sum(sum((test - q{7}.a).^2))/(numtest);

testerror_xent = -(1.0/samples)*sum(sum(test.*log(p{7}.a) + (1-test).*log(1-p{7}.a)));
testerror_xent2 = -(1.0/samples)*sum(sum(test.*log(q{7}.a) + (1-test).*log(1-q{7}.a)));
test_recon = reshape(p{7}.a,[size(test,1) 28 28]);
test_recon2 = reshape(q{7}.a,[size(test,1) 28 28]);
test_reshaped = reshape(test, [size(test,1) 28 28]);
for i = 1:10
    ind = randi([1 samples],1);
    temp = reshape(test_reshaped(ind,:,:),[28 28]);
    temp1 = reshape(test_recon(ind,:,:),[28 28]);
    temp_sgd = reshape(test_recon2(ind,:,:),[28 28]);
    temp_pca = reshape(test_recon_pca(ind,:,:),[28 28]);
    temp2 = [temp ;temp_sgd; temp1; temp_pca];
    output = [output temp2];
end
imshow(output);

