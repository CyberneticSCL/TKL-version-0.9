function [SVM] = findP(SVM,Kernel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [SVM,Obj] = findP(SVM,Kernel) function takes a support vector machine,
% object and a kernel object and finds an update to the P matrix of the
% kernel function.
% 
% INPUT
% SVM:    SVM object.
% Kernel: An internal kernel object used for quickly calculating the kernel matrix.
%
% OUTPUT
% SVM:    Optimized SVM (for given kernel function).
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - findP
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Calculate C
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
        cTemp(i,j) = -.5.*(w)'*kTemp*(w);
    end
end
C{1} = .5.*(cTemp+cTemp'); % Ensure C is symmetric.

%% Analytical Solution
[V,D] = eig(C{1}); % Calculate eigenvalues and eigenvectors
V = V(:,find(min(diag(D)) == diag(D))); % Select the eigenvector that corresponds to the minimum eigenvalue
P= length(V).*V*V'; % Calculate optimal P matrix

%% Calculate Dual Gap
[K] = makeK(SVM,Kernel,P); % Make kernel matrix
if strcmp(SVM.type,'Regression')
    ObjNew = -.5*SVM.Params.alpha'*K(SVM.Params.pos,SVM.Params.pos)*SVM.Params.alpha-SVM.Params.epsilon*sum(abs(SVM.Params.alpha))+sum(SVM.y(SVM.Params.pos)*SVM.Params.alpha); % Objective for optimal P with given alpha
elseif strcmp(SVM.type,'Classification')
    ObjNew = sum(SVM.Params.alpha)-.5*w'*K(SVM.Params.pos,SVM.Params.pos)*w; % Objective for optimal P with given alpha
end
SVM.Opt.dualGap2(end+1) = min(SVM.Opt.dualGap2(end),SVM.Opt.Obj(end)-ObjNew);

%% Update P
Pold = SVM.Params.P; % Previous P matrix
if length(SVM.Opt.StepLength) == 1
    eta = 1;
else
    eta = mean(SVM.Opt.StepLength); % Step length
end
etaMin = eta.*1e-3;

SVM.Params.P = Pold + eta.*(P-Pold);
[SVM,Obj] = findAlpha(SVM,Kernel);  % Update alpha
go = true; stepDecrease = 5;
while (Obj >= SVM.Opt.Obj(end)) & go
    eta = eta./stepDecrease; % Decrease step length
    if eta <= etaMin % Minimum step length
        eta = etaMin;
        go  = false; % ends iteration
        SVM.Params.P = Pold + eta.*(P-Pold);
        [SVM,Obj] = findAlpha(SVM,Kernel);  % Update alpha
    else
        SVM.Params.P = Pold + eta.*(P-Pold);
        [SVM,Obj] = findAlpha(SVM,Kernel);  % Update alpha
    end
end

SVM.Opt.l(end+1)       = max(SVM.Opt.l(end),SVM.Opt.Obj(end) + sum(sum((P-Pold).*C{1})));
SVM.Opt.dualGap(end+1) = Obj - SVM.Opt.l(end); % Duality Gap


SVM.Opt.Obj(end+1) = Obj; % Update objective value
SVM.Opt.diff = abs(SVM.Opt.Obj(end)-SVM.Opt.Obj(end-1))./abs(SVM.Opt.Obj(end) + SVM.Opt.Obj(end-1))*200; % Update the percentage difference in the Objective function
SVM.Opt.StepLength(end+1) = eta;
end

