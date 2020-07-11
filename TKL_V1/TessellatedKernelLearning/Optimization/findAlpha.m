function [SVM,Obj] = findAlpha(SVM,Kernel)
%FINDALPHA Summary of this function goes here
%   Detailed explanation goes here

if strcmp(SVM.type,'Classification')
    %%%%%%%%%%%%%%%%%% LIBSVM %%%%%%%%%%%%%%%%%%%%
    [SVM] = makeK(SVM,Kernel);

    K1 = [(1:length(SVM.y))', SVM.Params.K]; % include sample serial number as first column
    model = svmtrain(SVM.y', K1, ['-t 4 -s 0 -c ', num2str(SVM.Params.C),' -q 1']);

    Obj = -model.obj;
    SVM.Params.pos = model.SVs; SVM.Params.alpha = abs(model.sv_coef);
    SVM.Params.b = -model.rho;
elseif strcmp(SVM.type,'Regression')
    %%%%%%%%%%%%%%%%%% LIBSVM %%%%%%%%%%%%%%%%%%%%
    [SVM] = makeK(SVM,Kernel);

    K1 = [(1:length(SVM.y))', SVM.Params.K]; % include sample serial number as first column
    model = svmtrain(SVM.y', K1, ['-t 4 -s 3 -e ', num2str(SVM.Params.epsilon), ' -c ', num2str(SVM.Params.C),' -q 1']);

    Obj = -model.obj;
    SVM.Params.pos = model.SVs; SVM.Params.alpha = model.sv_coef;
    SVM.Params.b = -model.rho;
else
    error('No type of SVM specified')
end
end

