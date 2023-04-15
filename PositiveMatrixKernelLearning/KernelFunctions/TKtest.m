function [K] = TKtest(x,y,Z1,Z2,a,b,numx,numtest,P)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [K] = TKtest(x,y,Z1,Z2,a,b,num) function takes two matrices of inputs,
% as well as monomial basis of the inputs (Z1,Z2) and a lower (a) and upper
% (b) bound over which we integrate, the number of training inputs (numx), 
% the number of test inputs (numtest) and a matrix P that parameterizes the
% TK kernel function.  Computes the test kernel matrix for a TK kernel.
% 
% INPUT
% x:       Matrix of inputs to precompute portions of the kernel matrix.
% y:       Matrix of inputs to precompute portions of the kernel matrix.
% Z1:      Monomial basis of the training inputs.
% Z2:      Monomial basis of the test inputs.
% a:       Lower bound of integration for the TK kernel.
% b:       Upper bound of integration for the TK kernel.
% numx:    The number of training inputs.
% numtest: The number of testing inputs.
% P:       The P matrix which parameterizes the TK kernel function.
%
% OUTPUT
% K: The test kernel matrix.  
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - TKtest
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[dim] = size(x,1);
for n = 1:dim
    kTemp = kron(x(n,:)',ones(1,numtest));
    kTemp(:,:,2) = kron(y(n,:)',ones(1,numx))';
    if n == 1
        Ktemp{1,1} = b(n) - max(kTemp,[],3);
        Ktemp{1,2} = b(n) - kTemp(:,:,1);
        Ktemp{2,1} = b(n) - kTemp(:,:,2);
    else
        Ktemp{1,1} = Ktemp{1,1}.*(b(n) - max(kTemp,[],3));
        Ktemp{1,2} = Ktemp{1,2}.*(b(n) - kTemp(:,:,1));
        Ktemp{2,1} = Ktemp{2,1}.*(b(n) - kTemp(:,:,2));
    end
end
clear kTemp;
Ktemp{1,2} = Ktemp{1,2} - Ktemp{1,1};
Ktemp{2,1} = Ktemp{2,1} - Ktemp{1,1};
Ktemp{2,2} = prod(b-a) - Ktemp{1,1} - Ktemp{1,2} - Ktemp{2,1};

%% Compute the test kernel matrix
K = zeros(numx,numtest); q = length(P);
for i = 1:2
    for j = 1:2
        K = K+Ktemp{i,j}.*(Z1'*P(1+q/2*(i-1):q/2*i,1+q/2*(j-1):q/2*j)*Z2);
    end
end

    %% TK Kernel function subroutine
    function [val] = tTK(a,b,c)
        val = prod( (b.^c)./c-(a.^c)./c , 2); % subroutine to calculate the kernel function
    end
end