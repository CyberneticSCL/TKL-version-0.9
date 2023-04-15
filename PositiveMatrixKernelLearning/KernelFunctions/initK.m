function [K] = initK(x,a,b,num,kernel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [K] = initK(x,y,a,b,num) function takes two matrices of inputs, as well as
% a lower (a) and upper (b) bound over which we integrate and the number of
% inputs.
% 
% INPUT
% x:   Matrix of inputs to precompute portions of the kernel matrix.
% y:   Matrix of inputs to precompute portions of the kernel matrix.
% a:   Lower bound of integration for the kernel.
% b:   Upper bound of integration for the kernel.
% num: The number of inputs (equivalent to the size of the Kernel matrix).
%
% OUTPUT
% K: Precomputation of parts of the kernel matrix.  
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - initK
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcat(kernel,'TK')
    dim = size(x,1);
    for n = 1:dim
        kTemp = kron(x(n,:),ones(num,1));
        kTemp(:,:,2) = kTemp';
        if n == 1
            K{1,1} = b(n) - max(kTemp,[],3);
            K{1,2} = b(n) - kTemp(:,:,1);
            K{2,1} = b(n) - kTemp(:,:,2);
        else
            K{1,1} = K{1,1}.*(b(n) - max(kTemp,[],3));
            K{1,2} = K{1,2}.*(b(n) - kTemp(:,:,1));
            K{2,1} = K{2,1}.*(b(n) - kTemp(:,:,2));
        end
    end
    clear kTemp;
    K{1,2} = K{1,2} - K{1,1};
    K{2,1} = K{2,1} - K{1,1};
    K{2,2} = prod(b-a) - K{1,1} - K{1,2} - K{2,1};
else
    error('That kernel type has not been included.')    
end
end