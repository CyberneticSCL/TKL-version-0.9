function [K] = makeK(SVM,Kernel,P)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [K] = makeK(SVM,Kernel) function takes a support vector machine object, 
% and a Kernel object to generate the kernel matrix.
% 
% INPUT
% SVM:    SVM Object
% Kernel: An internal kernel object used for quickly calculating the kernel matrix.
% P:      An optional P argument can be given to be used instead of the P matrix saved in SVM.
%
% OUTPUT
% K: The kernel matrix.  
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - makeK
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcat(SVM.Params.kernel,'TK')
    if nargin == 2
        P = SVM.Params.P; % P matrix of the Positive Matrix Kernel Function
    end
    tempK = Kernel.K; % Kernel object for fast computation of the kernel matrix.
    q = SVM.Params.q; % Dimension of the P matrix
    K = zeros(size(tempK{1})); % Initialization of the Kernel matrix
    for i = 1:2
        for j = 1:2
            K = K+tempK{i,j}.*(Kernel.Z'*P(1+q/2*(i-1):q/2*i,1+q/2*(j-1):q/2*j)*Kernel.Z); % Kernel matrix evaluation
        end
    end
    K = .5.*(K+K');
else
    error('That kernel type has not been included.')
end
end

