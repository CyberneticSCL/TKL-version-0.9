function [Kij] = initiate_grad_P(Kernel, i, j, pos)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [SVM,Obj] = initiate_grad_P(Kernel, i, j, pos) function takes a kernel 
% object and finds the dual gap
% 
% INPUT 
% Kernel: An internal kernel object used for quickly calculating the kernel matrix.
% i,j: indexes
% pos: position of non-zero alphas;
%
% OUTPUT
% Kij:   Gradient of Phi by P_ij.
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - initiate_grad_P
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

q = size(Kernel.Z, 1); % Dimension of the P matrix
tempK = Kernel.K; % Kernel object for fast computation of the kernel matrix.
K = zeros(size(tempK{1})); % Initialization of the Kernel matrix
P = zeros(q, q);

index1 = 1;
index2 = 1;
if i > q
    index1 = 2;
    i = i-q;
end
if j > q
    index2 = 2;
    j = j-q;
end
P(i,j) = 1;

Kij = tempK{index1,index2}.*(Kernel.Z'*P*Kernel.Z); % Kernel matrix evaluation
Kij = Kij(pos, pos);

end