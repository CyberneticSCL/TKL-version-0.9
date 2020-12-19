function [SVM] = scaling(SVM,X)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [SVM] = Scaling(SVM,X) function takes a support vector machine object, 
% and scales the training data to fit within the hypercube [0,1]^n, then
% saves the scaled data and the scaling factors.
% 
% INPUT
% SVM: TK SVM object.
% X:   Unscaled training data.
%
% OUTPUT
% SVM: TK SVM object (now has scaled training data)  
% 
% NOTES:
% For support, contact M. Peet, Arizona State University at mpeet@asu.edu
% or B.K. Colbert at brendon.colbert@asu.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TKL - scaling
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

