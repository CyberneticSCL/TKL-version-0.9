function [K] = makeK(SVM,Kernel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [K] = makeK(SVM,Kernel) function takes a support vector machine object, 
% and a Kernel object to generate the kernel matrix.
% 
% INPUT
% SVM:    Optimized TK SVM (output of TKL function)
% Kernel: An internal kernel object used for quickly calculating the kernel matrix.
%
% OUTPUT
% K: The kernel matrix.  
% 
% NOTES:
% For support, contact M. Peet, Arizona State University at mpeet@asu.edu
% or B.K. Colbert at brendon.colbert@asu.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TKL - makeK
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


P = SVM.Params.P; % P matrix of the TK Kernel
tempK = Kernel.K; % Kernel object for fast computation of the kernel matrix.
q = SVM.Params.q; % Dimension of the P matrix
K = zeros(size(tempK{1})); % Initialization of the Kernel matrix
for i = 1:2
    for j = 1:2
        K = K+tempK{i,j}.*(Kernel.Z'*P(1+q/2*(i-1):q/2*i,1+q/2*(j-1):q/2*j)*Kernel.Z); % Kernel matrix evaluation
    end
end
end

