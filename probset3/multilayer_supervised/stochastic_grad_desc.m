function [ W, error ] = stochastic_grad_desc(func, W0, alpha, N, train_X, train_y, test_X, test_y, ei)
    
    % Determine size of the dataset 
    [~, m] = size(train_X);
    W = W0;
    error = zeros(1, N);
    disp('error');
    
    % Iterate until max iterations, N, reached
    for iteration = 1:N
        % Shuffle the examples on each iteration
        perm = randperm(m);
        train_X = train_X(:, perm);
        train_y = train_y(perm);
        
        % Run over the dataset, randomly permuted for each iteration
        for j = 1:m
            [~, g] = func(W, ei, train_X(:, j), train_y(j), false);
            W = W - alpha * g;
        end
        
        % Predict performance with current parameter set
        [~, ~, pred] = func(W, ei, test_X, test_y, true);
        [~,pred] = max(pred);
        error(iteration) = 1 - mean(pred == test_y);
        disp(error(iteration));
    end
end

