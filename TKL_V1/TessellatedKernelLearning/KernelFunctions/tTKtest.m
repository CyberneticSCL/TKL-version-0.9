function [K] = tTKtest(x,y,Z1,Z2,a,b,numx,numtest,P)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [K] = tTKtest(x,y,Z1,Z2,a,b,num) function takes two matrices of inputs,
% as well as monomial basis of the inputs (Z1,Z2) and a lower (a) and upper
% (b) bound over which we integrate, the number of training inputs (numx), 
% the number of test inputs (numtest) and a matrix P that parameterizes the
% TK kernel function.  Computes the test kernel matrix.
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
% NOTES:
% For support, contact M. Peet, Arizona State University at mpeet@asu.edu
% or B.K. Colbert at brendon.colbert@asu.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TKL - initTK
%
% Copyright (C)2019  M. Peet, B.K. Colbert
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% If you modify this code, document all changes carefully and include date
% authorship, and a brief description of modifications
%
% Initial coding MMP, BKC  - 12_15_2020
%
%% Calculate the maximum values between the training input matrices and testing input matrices.
[n,dim] = size(x); q = length(P);
for j = 1:dim
    m(:,j) = max([x(:,j),y(:,j)]')';
end

%% Compute the integral portion of the Kernel matrix.
Ktemp{1,1} = reshape(tTK(m,b,1),numx,numtest);
Ktemp{2,1} = reshape(tTK(y,b,1) - tTK(m,b,1),numx,numtest);
Ktemp{1,2} = reshape(tTK(x,b,1) - tTK(m,b,1),numx,numtest);
Ktemp{2,2} = reshape(tTK(a,b,1) - tTK(m,b,1) - tTK(x,b,1) - tTK(y,b,1),numx,numtest);

%% Compute the test kernel matrix
K = zeros(numx,numtest);
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