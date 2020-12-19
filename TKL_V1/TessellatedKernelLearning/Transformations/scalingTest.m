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

%% Scale each dimension of the data
dim = size(X,1); % Dimension of the training data
for n = 1:dim
    X(n,:) = (X(n,:) - SF(n,1))./(SF(n,2) - SF(n,1));
end

end

