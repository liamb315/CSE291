p = activate(test,unfold_stack(stack_rbm_2d));
q = activate(test,stack_sgd_2d);

samples = size(test,1);
[coeff,score,latent] = pca(train);
test_decom = test*coeff(:,1:2);
test_recon_pca = test_decom*(coeff(:,1:2)');

output = p{4}.a;
output2 = q{4}.a;
output3 = test_recon_pca;
filename_lab = './MNIST/train-labels.idx1-ubyte';
labels = loadMNISTLabels(filename_lab);
numtrain = 20000;
numtest = 5000;
train_lab = labels(1:numtrain,:);
test_lab = labels(numtrain+1:numtrain + numtest,:);

combo = [output test_lab];
[values, order] = sort(combo(:,3));
sortcombo = combo(order,:);
sortcombo(:,3) = sortcombo(:,3)/10.0 + 0.1;
scatter(sortcombo(:,1),sortcombo(:,2),8,sortcombo(:,3))
axis off;
colormap(jet);
saveas(gcf, 'rbm_only.png');

combo = [output2 test_lab];
[values, order] = sort(combo(:,3));
sortcombo = combo(order,:);
sortcombo(:,3) = sortcombo(:,3)/10.0 + 0.1;
scatter(sortcombo(:,1),sortcombo(:,2),8,sortcombo(:,3))
axis off;
colormap(jet);
saveas(gcf, 'rbm_backprop.png');

combo = [output3 test_lab];
[values, order] = sort(combo(:,3));
sortcombo = combo(order,:);
sortcombo(:,3) = sortcombo(:,3)/10.0 + 0.1;
scatter(sortcombo(:,1),sortcombo(:,2),8,sortcombo(:,3))
axis off;
colormap(jet);
saveas(gcf, 'pca.png');

