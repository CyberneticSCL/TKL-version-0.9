function [Kt] = makeKtest(SVM,xTest)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [K] = makeKtest(SVM,xTest) function takes an optimized support vector
% machine object, and test inputs to generate a kernel matrix between the 
% training inputs and the test inputs.
% 
% INPUT
% SVM:   Optimized SVM object (output of PMKL)
% xTest: Test inputs.
%
% OUTPUT
% Kt: The test kernel matrix.  
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - makeKtest
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xtrain = SVM.x(:,SVM.Params.pos);
[dimx,numx] = size(xtrain); numtest = size(xTest,2);

if strcmp(SVM.Params.kernel,'TK')
    Kt = TKtest(xtrain,xTest,monomials(xtrain,SVM.Params.degree),monomials(xTest,SVM.Params.degree),SVM.Params.Lower,SVM.Params.Upper,numx,numtest,SVM.Params.P)';
else
    error('That kernel type has not been included.')
end
end

