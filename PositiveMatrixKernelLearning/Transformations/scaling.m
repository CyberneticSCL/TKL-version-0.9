function [SVM] = scaling(SVM,X)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [SVM] = Scaling(SVM,X) function takes a support vector machine object, 
% and scales the training data to fit within the hypercube [0,1]^n, then
% saves the scaled data and the scaling factors.
% 
% INPUT
% SVM: SVM object.
% X:   Unscaled training data.
%
% OUTPUT
% SVM: SVM object (now has scaled training data)  
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - scaling
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Scale each dimension of the data to be within [0,1].
dim = size(X,1); % Dimension of the training data
for n = 1:dim
    scaleFactor(n,:) = [min(X(n,:)),max(X(n,:))]; % Save scaling factor
    if diff(scaleFactor(n,:)) == 0 % Condition applies if all data is the same value
        scaleFactor(n,:) = [-abs(min(X(n,:))),abs(min(X(n,:)))];
    end
    X(n,:) = (X(n,:) - scaleFactor(n,1))./(scaleFactor(n,2) - scaleFactor(n,1)); % Scale the data
end
SVM.x = X; % Save scaled data
SVM.scaleFactor = scaleFactor; % Save scaling factors

end

