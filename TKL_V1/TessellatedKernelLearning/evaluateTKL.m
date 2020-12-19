function [yPred] = evaluateTKL(SVM,inputs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [params] = evaluateTKL(SVM,xTest) function takes an optimized support
% vector machine, and a set of inputs to be evaluated.
% 
% INPUT
% SVM:    Optimized TK SVM (output of TKL function)
% inputs: Inputs where we want to predict an output.
%
% OUTPUT
% yPred: Predicted output for each input value.  
% 
% NOTES:
% For support, contact M. Peet, Arizona State University at mpeet@asu.edu
% or B.K. Colbert at brendon.colbert@asu.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TKL - evaluateTKL
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

xTest = scalingTest(inputs,SVM.scaleFactor); % Scale testing data as training data was
Kt = makeKtest(SVM,xTest); % Create kernel matrix

if strcmp(SVM.type,'Classification') 
    yPred = sign(Kt*(SVM.y(SVM.Params.pos)'.*SVM.Params.alpha) + SVM.Params.b); % Generate predictions for binary classification
elseif strcmp(SVM.type,'Regression')
    yPred = Kt*SVM.Params.alpha + SVM.Params.b; % Generate predictions for regression problems
end

end

