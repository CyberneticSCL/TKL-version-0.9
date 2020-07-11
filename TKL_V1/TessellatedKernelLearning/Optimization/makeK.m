function [SVM] = makeK(SVM,Kernel)
%MAKEK Summary of this function goes here
%   Detailed explanation goes here

Kfast = Kernel.K; P = SVM.Params.P;
K = zeros(size(Kfast{1}));
for i = 1:length(Kfast)
    for j = 1:length(Kfast)
        K = K+P(i,j).*Kfast{i,j};
    end
end
SVM.Params.K = K;
end

