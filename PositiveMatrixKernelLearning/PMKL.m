function [SVM] = PMKL(x,y,type,C,params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM = PMKL(x,y,type,C,params) function maps inputs x, to outputs
% y where the "type" can be classification or regression type mapping, C is
% a regularization parameter and params holds extra parameter values.
% 
% INPUT
% x:      Inputs to be mapped to outputs y.
% y:      Outputs.
% type:   Classification for binary mappings and regression for real valued mappings.
% C:      Regularization parameter, smaller values lead to mappings that are more general.
% params: Parameter object that holds additional parameters.
%
% OUTPUT
% SVM: A fit SVM with an optimal Positive Matrix Kernel Function.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - PMKL
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input handling and default values
if nargin < 2
    error('Not enough input arguments')
elseif nargin == 2
    C = 10;
    params = paramsTKL();
    if length(unique(y)) == 2
        type = 'Classification';
    else
        type = 'Regression';
    end
elseif nargin == 3
    C=10;
    params = paramsTK();
elseif nargin == 4
    params = paramsTK();
elseif nargin > 6
    error('Too many input arguments')
end

%% Set up Kernel Function
if size(y,1)>1 & size(y,2) == 1 % Ensure output is the right dimension
    SVM.y = y';
elseif size(y,1) == 1 & size(y,2) > 1 
    SVM.y = y;
else
    error('y must be a vector')
end

if length(y) == size(x,1) % Ensure input is the right dimension
    SVM.xOld = x';
elseif length(y) == size(x,2)
    SVM.xOld = x;
else
    error('x and y must contain the same number of points')
end
SVM = scaling(SVM,SVM.xOld); % Scale the data to ensure numerical stability


%% Save Kernel Variables
SVM.Params.kernel = params.kernel; SVM.Params.degree = params.degree;

%% Kernel Matrix set up
[dim,num] = size(SVM.x); % Dimension and number of inputs
SVM.Params.Lower = min(SVM.x')-params.bound; % Lower bounds of integration
SVM.Params.Upper = max(SVM.x')+params.bound; % Upper bounds of integration

%% Initialize Kernel Matrix
Kernel.K = initK(SVM.x, SVM.Params.Lower, SVM.Params.Upper,num, SVM.Params.kernel); % Initialize Kernel Matrix to speed up kernel matrix computation
Kernel.Z = monomials(SVM.x, SVM.Params.degree); % Initialize matrix of monomials of our input data
SVM.Params.q = 2.*size(Kernel.Z,1);

%% Set up TKL
q = SVM.Params.q; SVM.Params.P = eye(q); % Initialize P matrix
Eps = params.epsilon; tol = params.tol; maxit = params.maxit; SVM.Params.C = C; SVM.Params.epsilon = Eps;
SVM.Opt.StepLength = []; SVM.Opt.dualGap = []; SVM.Opt.dualGap2 = [inf]; SVM.Opt.l(1) = -inf; 
SVM.Opt.Obj = -inf; % Initialize Objective value
SVM.Opt.StepLength(1) = 1; SVM.Opt.T = [];
SVM.Params.tol = tol;

% Record SVM type
if (strcmp(type,'Classification') | strcmp(type,'classification') | strcmp(type,'C') | strcmp(type,'c'));
    SVM.type = 'Classification';    
else(strcmp(type,'Regression') | strcmp(type,'regression') | strcmp(type,'R') | strcmp(type,'r'));
    SVM.type = 'Regression';
end

%% TKL Algorithm
tic;
[SVM,SVM.Opt.Obj(2)] = findAlpha(SVM,Kernel); % Finds values for alpha
SVM.Opt.T(end+1) = toc;
go = true; iter = 0; SVM.Opt.diff = abs(SVM.Opt.Obj(end)-SVM.Opt.Obj(end-1))./abs(SVM.Opt.Obj(end) + SVM.Opt.Obj(end-1))*200; % Calculates the percent difference between objective values
fprintf('Iteration   |  Objective   |       Dual Gap      | \n')
fprintf('------------+--------------+---------------------| \n')
while go
    iter = iter+1;
    tic;
    SVM = findP(SVM,Kernel); % Updates the P matrix which parameterizes the Positive Matrix Kernel Function.
    SVM.Opt.T(end+1) = toc;
    if (iter > maxit) | (min(SVM.Opt.dualGap2(end),SVM.Opt.dualGap(end)) < SVM.Params.tol) | (SVM.Opt.StepLength(end) < 1/100*SVM.Params.tol) % Quits if maxit > current iteration or if the objective function change was too small.
        go = false;
    end
    fprintf('%10d  |  %1.4e  |  %1.4e \n', iter, SVM.Opt.Obj(end), min(SVM.Opt.dualGap2(end)));
end

