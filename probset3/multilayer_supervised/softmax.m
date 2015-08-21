function [P]  = softmax(z)
    numerator = exp(z);
    denominator = sum(numerator);
    P = bsxfun(@rdivide, numerator, denominator);
end