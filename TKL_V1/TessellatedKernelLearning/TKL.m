function [SVM] = TKL(x,y,type,C,params)
%TKL Finds the optimal TK kernel function.
%   First two inputs are the inputs x, and outputs y
%   Third input type should be regression or classification
%   Fourth input 'C' is the penalty constraint and must be greater than 0.
%   Fifth input 'Params' contains extra kernel/svm parameters.
%   Minimum inputs are x, y, and type.

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
    params = paramsTKL();
elseif nargin == 4
    params = paramsTKL();
elseif nargin == 5
    params2 = paramsTKL();
    try 
        deg = params.deg
    catch
        deg = params2.deg;
    end
    try
        epsilon = params.epsilon;
    catch
        epsilon = params2.epsilon;
    end
    try
        maxit = params.maxit;
    catch
        maxit = params2.maxit;
    end
    try
        tol = params.tol;
    catch
        tol = parasm2.tol;
    end
elseif nargin > 5
    error('Too many input arguments')
end

%% Set up Kernel Function
if size(y,1)>1 & size(y,2) == 1
    SVM.y = y';
elseif size(y,1) == 1 & size(y,2) > 1
    SVM.y = y;
else
    error('y must be a vector')
end

if length(y) == size(x,1)
    SVM.xOld = x';
elseif length(y) == size(x,2)
    SVM.xOld = x;
else
    error('x and y must contain the same number of points')
end
SVM = Scaling(SVM,SVM.xOld);


[dim,num] = size(SVM.x);
SVM.Params.delta = monomialIndex(dim,params.deg);
SVM.Params.gamma = SVM.Params.delta.*0;
SVM.Params.Lower = min(SVM.x')-params.bound;
SVM.Params.Upper = max(SVM.x')+params.bound;

comb = num^2;
Kernel.y = zeros(dim,comb); Kernel.x = Kernel.y;
pos1 = 1; add = num-1;
for n = 1:num
    Kernel.y(:,pos1:pos1+add) = SVM.x;
    Kernel.x(:,pos1:pos1+add) = kron(SVM.x(:,n),ones(1,add+1));
    pos1 = pos1+add+1;
end
Kernel.K = gTKfast(Kernel.x',Kernel.y',SVM.Params.Lower,SVM.Params.Upper,SVM.Params.delta,SVM.Params.gamma,num);

SVM.Params.KernelType = 'TK';
SVM.Params.degree     = params.deg;
SVM.Params.q          = 2.*size(SVM.Params.delta,1);

%% Set up TKL
Eps   = params.epsilon;
tol   = params.tol;
maxit = params.maxit;

q = 2.*size(SVM.Params.delta,1);
SVM.Params.P = eye(q); SVM.Params.C = C; SVM.Params.epsilon = Eps; SVM.Opt.Obj = -inf;

if (strcmp(type,'Classification') | strcmp(type,'classification') | strcmp(type,'C') | strcmp(type,'c'))
    SVM.type = 'Classification';    
else(strcmp(type,'Regression') | strcmp(type,'regression') | strcmp(type,'R') | strcmp(type,'r'))
    SVM.type = 'Regression';
end

[SVM,SVM.Opt.Obj(2)] = findAlpha(SVM,Kernel); % Finds values for alpha
go = true; iter = 0; SVM.Opt.diff = abs(SVM.Opt.Obj(end)-SVM.Opt.Obj(end-1))./abs(SVM.Opt.Obj(end) + SVM.Opt.Obj(end-1))*200; % Calculates the percent difference between objective values
fprintf('Iteration   |  Objective   |   Per. Difference   | \n')
fprintf('------------+--------------+---------------------| \n')
while go
    iter = iter+1;
    SVM = findP(SVM,Kernel); % Updates the P matrix which parameterizes the TK kernel function.
    if iter > maxit | SVM.Opt.diff < tol % Quits if maxit > current iteration or if the objective function change was too small.
        go = false;
    end
    fprintf('%10d  |  %1.4e  |  %1.4e \n', iter, SVM.Opt.Obj(end), SVM.Opt.diff);
end
SVM = findAlpha(SVM,Kernel); % Calculates alpha for final P matrix value.

