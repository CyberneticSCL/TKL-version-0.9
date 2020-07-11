function [yPred,Kt] = evaluateSVM(SVM,xTest)
%PREDICTPK Predicts the output of a SVM using a TK kernel function
%   First input - Trained SVM
%   Second input - Training Data

xTest = ScalingTest(xTest,SVM.ScaleFactor); % Scale testing data as training data was
Kt = makeTKTest(SVM,xTest); % Create kernel matrix

if strcmp(SVM.type,'Classification') 
    yPred = sign(Kt*(SVM.y(SVM.Params.pos)'.*SVM.Params.alpha) + SVM.Params.b); % Generate predictions
elseif strcmp(SVM.type,'Regression')
    yPred = Kt*SVM.Params.alpha + SVM.Params.b; % Generate predictions
end

end

