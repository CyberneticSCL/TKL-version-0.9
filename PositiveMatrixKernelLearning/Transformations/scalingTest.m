function [ X ] = scalingTest( X, SF )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X = scalingTest(X,SF) function takes an input X and returns scaled input 
% data scaled using the values in SF.
% 
% INPUT
% X:  Unscaled inputs.
% SF: Matrix of scaling factors.
%
% OUTPUT
% X: Scaled input data.  
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - scaling
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Scale each dimension of the data
dim = size(X,1); % Dimension of the training data
for n = 1:dim
    X(n,:) = (X(n,:) - SF(n,1))./(SF(n,2) - SF(n,1) + 1.e-6);
end

end

