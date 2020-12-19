function [Kt] = makeKtest(SVM,xTest)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [K] = makeKtest(SVM,xTest) function takes an optimized support vector
% machine object, and test inputs to generate a kernel matrix between the 
% training inputs and the test inputs.
% 
% INPUT
% SVM:   Optimized TK SVM (output of TKL function)
% xTest: Test inputs.
%
% OUTPUT
% Kt: The test kernel matrix.  
% 
% NOTES:
% For support, contact M. Peet, Arizona State University at mpeet@asu.edu
% or B.K. Colbert at brendon.colbert@asu.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TKL - makeKtest
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
xtrain = SVM.x(:,SVM.Params.pos);
[dimx,numx] = size(xtrain); numtest = size(xTest,2);

y = zeros(dimx,numx*numtest); x = y;
pos1 = 1; add = numx-1;
for n = 1:numtest
    y(:,pos1:pos1+add) = kron(xTest(:,n),ones(1,numx));
    x(:,pos1:pos1+add) = xtrain;
    pos1 = pos1+add+1;
end

Kt = tTKtest(x',y',monomials(xtrain,SVM.Params.degree),monomials(xTest,SVM.Params.degree),SVM.Params.Lower,SVM.Params.Upper,numx,numtest,SVM.Params.P)';
end

