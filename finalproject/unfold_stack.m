function [stack_unfold] = unfold_stack(stack)
numlayers = 2*length(stack);
stack_unfold = cell(1, numlayers);
for l = 1:numlayers
    if l > numlayers/2
        stack_unfold{l}.W = stack{numlayers -l + 1}.W';
        stack_unfold{l}.b = stack{numlayers -l + 1}.a;
    else
        stack_unfold{l}.W = stack{l}.W;
        stack_unfold{l}.b = stack{l}.b;
    end
end
end