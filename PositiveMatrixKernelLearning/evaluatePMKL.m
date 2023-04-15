function [yPred] = evaluatePMKL(SVM,inputs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [params] = evaluatePMKL(SVM,xTest) function takes an optimized support
% vector machine, and a set of inputs to be evaluated.
% 
% INPUT
% SVM:    SVM Object (output of PMKL function)
% inputs: Inputs where we want to predict an output.
%
% OUTPUT
% yPred: Predicted output for each input value.  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - evaluatePMKL
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xTest = scalingTest(inputs,SVM.scaleFactor); % Scale testing data as training data was
Kt = makeKtest(SVM,xTest); % Create kernel matrix

if strcmp(SVM.type,'Classification') 
    yPred = sign(Kt*(SVM.y(SVM.Params.pos)'.*SVM.Params.alpha) + SVM.Params.b); % Generate predictions for binary classification
elseif strcmp(SVM.type,'Regression')
    yPred = Kt*SVM.Params.alpha + SVM.Params.b; % Generate predictions for regression problems
end

end

