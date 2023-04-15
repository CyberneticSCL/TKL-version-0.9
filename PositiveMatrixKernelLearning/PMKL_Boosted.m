function [SVM] = PMKL_Boosted(x,y,type,C,params)
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
% SVM: A fit SVM with a TK kernel.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - PMKL_Boosted
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning('off','all')
%% Input handling and default values
if nargin < 2
    error('Not enough input arguments')
elseif nargin == 2
    C = 10;
    params = paramsTK();
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
elseif nargin > 5
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
     xOld = x';
elseif length(y) == size(x,2)
     xOld = x;
else
    error('x and y must contain the same number of points')
end
SVM = scaling(SVM,xOld); % Scale the data to ensure numerical stability


%% Save Kernel Variables
SVM.Params.kernel = params.kernel; SVM.Params.degree = params.degree;

%% Kernel Matrix set up
[dim,num] = size(SVM.x); % Dimension and number of inputs
SVM.Params.Lower = min(SVM.x')-params.bound; % Lower bounds of integration
SVM.Params.Upper = max(SVM.x')+params.bound; % Upper bounds of integration
SVM.Params.Solver_runs = 0;
% SVM.Params.delta = monomialIndex(dim,params.deg);
% SVM.Params.gamma = SVM.Params.delta.*0;
SVM.Params.regul_alpha = params.regul_alpha;

%% Initialize Kernel Matrix
Kernel.K = initK(SVM.x,SVM.Params.Lower,SVM.Params.Upper,num,SVM.Params.kernel); % Initialize Kernel Matrix to speed up kernel matrix computation
Kernel.Z = monomials(SVM.x,params.degree); % Initialize matrix of monomials of our input data
SVM.Params.q = 2.*size(Kernel.Z,1);

%% Set up PMKL
q = SVM.Params.q; SVM.Params.P = eye(q); % Initialize P matrix
Eps = params.epsilon; tol = params.tol; maxit = params.maxit; SVM.Params.C = C; SVM.Params.epsilon = Eps;
SVM.Opt.StepLength = 1; SVM.Opt.dualGap = []; SVM.Opt.l = -inf; SVM.Opt.dualGap2 = [inf];
SVM.Opt.Obj(1) = -inf; SVM.Opt.T = []; % Initialize Objective value
SVM.Params.tol = tol;
% Record SVM type
if (strcmp(type,'Classification') | strcmp(type,'classification') | strcmp(type,'C') | strcmp(type,'c'));
    SVM.type = 'Classification';    
else(strcmp(type,'Regression') | strcmp(type,'regression') | strcmp(type,'R') | strcmp(type,'r'));
    SVM.type = 'Regression';
end

c = clock;
SVM.Opt.time = c(end-2)*3600 + c(end-1)*60 + c(end);
SVM.Opt.real_Obj = SVM.Opt.Obj(end);
iter = 0;
restart_iter = params.restart_iter;
% fprintf('Iteration   |  Objective   |  Difference |\n')
% fprintf('------------+--------------+-------------|\n')
go = true;
%% Initial parameters
restart_index = 0; 
mu = 100;
[SVM,SVM.Opt.Obj(2)] = findAlpha(SVM,Kernel); % Finds values for alpha
SVM.Opt.diff = abs(SVM.Opt.Obj(end)-SVM.Opt.Obj(end-1))./abs(SVM.Opt.Obj(end) + SVM.Opt.Obj(end-1))*200; % Calculates the percent difference between objective values

%% Preprocessing
%%%%%%%% ------------+--------------+-------------+
% fprintf('===========Frank-Wolfe Algorithm==========\n');

% while go
%     iter = iter+1;
%     SVM = findP(SVM,Kernel); % Updates the P matrix which parameterizes the kernel function.
%     if iter > maxit | SVM.Opt.diff < 1.e-5% Quits if maxit > current iteration or if the objective function change was too small.
%         go = false;
%     end
% %     fprintf('%10d  |  %1.4e  |  %1.4e \n', iter, SVM.Opt.Obj(end), SVM.Opt.diff);
% end
% go = true;
% if iter > maxit | SVM.Opt.diff < SVM.Params.tol % Quits if maxit > current iteration or if the objective function change was too small.
%     go = false;
% end
if go
    cTemp = zeros(SVM.Params.q);
    if strcmp(SVM.type,'Regression')
        w = SVM.Params.alpha; % w depends on the type of SVM
    elseif strcmp(SVM.type,'Classification')
        w = SVM.Params.alpha.*SVM.y(SVM.Params.pos)'; % w depends on the type of SVM
    end
    for i=1:2*size(Kernel.Z,1)
        for j = 1:2*size(Kernel.Z,1)
            n = (i > size(Kernel.Z,1)) + 1; m = (j > size(Kernel.Z,1)) + 1;
            k = i - (n-1)*size(Kernel.Z,1); l = j - (m-1)*size(Kernel.Z,1);
            kTemp = Kernel.K{n,m}(SVM.Params.pos,SVM.Params.pos).*(Kernel.Z(k,SVM.Params.pos)'*Kernel.Z(l,SVM.Params.pos));
            cTemp(i,j) = -(w)'*kTemp*(w);
        end
    end
    G_P  = .5.*(cTemp+cTemp'); % Ensure C is symmetric.
    Lyx = 0;
    q = size(Kernel.Z, 1);
    %% Estimation of Lipschitz constants
    for i = 1:(2*q)
        for j = i:(2*q)
            Kij = initiate_grad_P(Kernel, i, j, 1:length(SVM.y));
            if i == j
               mu = min(mu, min(eigs(Kij)));
            end
            if strcmp(SVM.type,'Classification')
                Lyx =Lyx + C*norm(diag(SVM.y)*Kij*diag(SVM.y), 2);
            else
                Lyx =Lyx + C*norm(initiate_grad_P(Kernel, i, j, 1:length(SVM.y)), 2);
            end
            break
        end
        break
    end
    mu = 0.0001;
    mu = (mu + SVM.Params.regul_alpha ); 
    Lxx = Lyx*q/C;
    Lyx = Lyx;
    SVM.Params.Cold = G_P ;
    tau0 = 100*0.5/(Lxx );
    sigma0 = 100* Lxx /Lyx^2;
    if strcmp(SVM.type,'Classification')
        tau0 = 100*tau0;
        sigma0 = 100*sigma0;
    end
    gamma_k = sigma0/tau0;
    tau_k = tau0;
    sigma_k = sigma0;
    go = true;  
    %% ALGORITHM 2.2 APD
    c = clock; 
    fprintf("===============APD Algorithm==============\n");
    [SVM] = main_algorithm(SVM, Kernel, mu, tau0, sigma0, maxit, 1);
    c1 = clock;
    SVM.execution_time = c1(end-2)*3600 + c1(end-1)*60 + c1(end) - c(end-2)*3600 - c(end-1)*60 - c(end);
end
[SVM,SVM.Opt.Obj(end+1)] = findAlpha(SVM,Kernel); % Calculates alpha for final P matrix value.
 
