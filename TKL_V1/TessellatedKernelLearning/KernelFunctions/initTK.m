function [K] = initTK(x,y,a,b,num)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [K] = initTK(x,y,a,b,num) function takes two matrices of inputs, as well as
% a lower (a) and upper (b) bound over which we integrate and the number of
% inputs.
% 
% INPUT
% x:   Matrix of inputs to precompute portions of the kernel matrix.
% y:   Matrix of inputs to precompute portions of the kernel matrix.
% a:   Lower bound of integration for the TK kernel.
% b:   Upper bound of integration for the TK kernel.
% num: The number of inputs (equivalent to the size of the Kernel matrix).
%
% OUTPUT
% K: Precomputation of parts of the kernel matrix.  
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
%% Calculate the maximum values between the input matrices.
[n,dim] = size(x);
for j = 1:dim
    m(:,j) = max([x(:,j),y(:,j)]')';
end

%% Initialize the integral portion of the Kernel matrix.
for i = 1:2
    for j = 1:2
        K{i,j} = {zeros(num,num)}; % Initialize kernel matrices
    end
end

%% Precompute the integral portion of the Kernel matrix.
K{1,1} = reshape(tTK(m,b,1),num,num);
K{1,2} = reshape(tTK(y,b,1) - tTK(m,b,1),num,num);
K{2,1} = reshape(tTK(x,b,1) - tTK(m,b,1),num,num);
K{2,2} = reshape(tTK(a,b,1) - tTK(m,b,1) - tTK(x,b,1) - tTK(y,b,1),num,num);


    %% TK Kernel function subroutine
    function [val] = tTK(a,b,c)
        val = prod( (b.^c)./c-(a.^c)./c , 2); % subroutine to calculate the kernel function
    end
end