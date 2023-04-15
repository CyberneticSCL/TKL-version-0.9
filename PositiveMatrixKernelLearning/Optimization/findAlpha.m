function [SVM,Obj] = findAlpha(SVM,Kernel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [SVM,Obj] = findAlpha(SVM,Kernel) function takes a support vector
% machine object and a kernel object and optimizes the support vector
% machine with respect to the current kernel function.
% 
% INPUT
% SVM:    SVM object.
% Kernel: An internal kernel object used for quickly calculating the kernel matrix.
%
% OUTPUT
% SVM:    Optimized SVM (for specified kernel).
% Obj:    The objective value of the SVM with the current kernel.
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - findAlpha
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(SVM.type,'Classification')
    %%%%%%%%%%%%%%%%%% LIBSVM %%%%%%%%%%%%%%%%%%%%
    [SVM.Params.K] = makeK(SVM,Kernel); % Make kernel matrix

    K1 = [(1:length(SVM.y))', SVM.Params.K]; % include sample serial number as first column
    model = svmtrain(SVM.y', K1, ['-t 4 -s 0 -c ', num2str(SVM.Params.C),' -q 1']); % Use LibSVM to optimize SVM
    Obj = sum(abs(model.sv_coef))-.5*model.sv_coef'*SVM.Params.K(model.SVs,model.SVs)*model.sv_coef; % The objective value of the SVM
    SVM.Params.pos = model.SVs; SVM.Params.alpha = abs(model.sv_coef); % The position of the support vectors and their values
    SVM.Params.b = -model.rho; % The b parameter of the SVM
elseif strcmp(SVM.type,'Regression')
    %%%%%%%%%%%%%%%%%% LIBSVM %%%%%%%%%%%%%%%%%%%%
    [SVM.Params.K] = makeK(SVM,Kernel); % Make kernel matrix

    K1 = [(1:length(SVM.y))', SVM.Params.K]; % include sample serial number as first column
    model = svmtrain(SVM.y', K1, ['-t 4 -s 3 -e ', num2str(SVM.Params.epsilon), ' -c ', num2str(SVM.Params.C),' -q 1']);
    
    Obj = -.5*model.sv_coef'*SVM.Params.K(model.SVs,model.SVs)*model.sv_coef-SVM.Params.epsilon*sum(abs(model.sv_coef))+sum(SVM.y(model.SVs)*model.sv_coef); % The objective value of the SVM
    SVM.Params.pos = model.SVs; SVM.Params.alpha = model.sv_coef; % The position of the support vectors and their values
    SVM.Params.b = -model.rho; % The b parameter of the SVM
else
    error('No type of SVM specified')
end
end

