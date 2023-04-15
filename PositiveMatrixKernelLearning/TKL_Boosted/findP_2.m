function [SVM] = findP_2(SVM, Kernel, sigma, tau, theta, iter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [SVM,Obj] = findP_2(SVM,Kernel) function takes a support vector machine,
% object and a kernel object and finds an update to the P matrix of the
% kernel function.
% 
% INPUT
% SVM:    SVM object.
% Kernel: An internal kernel object used for quickly calculating the kernel matrix.
% sigma, tau, theta: Parameters for APD
% iter: current iteration
% OUTPUT
% SVM:    SVM object.
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - findP_2
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tol = SVM.Params.tol/10;
%% Set up SDPT3 Variables
cTemp = zeros(SVM.Params.q);
w = zeros(size(SVM.y')); 
if strcmp(SVM.type,'Regression')
    w = SVM.Params.alpha;
elseif strcmp(SVM.type,'Classification')
    w = SVM.Params.alpha.*SVM.y(SVM.Params.pos)';
end

%% Calculate C
for i=1:2*size(Kernel.Z,1)
    for j = 1:2*size(Kernel.Z,1)
        n = (i > size(Kernel.Z,1)) + 1; m = (j > size(Kernel.Z,1)) + 1;
        k = i - (n-1)*size(Kernel.Z,1); l = j - (m-1)*size(Kernel.Z,1);
        kTemp = Kernel.K{n,m}(SVM.Params.pos,SVM.Params.pos).*(Kernel.Z(k,SVM.Params.pos)'*Kernel.Z(l,SVM.Params.pos));
        cTemp(i,j) = -(w)'*kTemp*(w);
    end
end
C = .5.*(cTemp+cTemp');%Ensure C is symmetric.

grad_p = (theta).*C - (1 - theta).*SVM.Params.Cold;
%% Analytical Solution
Pold = SVM.Params.P;
if sigma == inf
    A = grad_p;
else
    A = Pold - grad_p*sigma;
end

%% Optimization line 3 APD 2.1
P = real(Algo_opt_trace(A, length(Pold), tol));
eta = 1.0;
SVM.Params.P = P;

[K] = makeK(SVM,Kernel,P); % Make kernel matrix
if strcmp(SVM.type,'Regression')
    ObjNew = -.5*SVM.Params.alpha'*K(SVM.Params.pos,SVM.Params.pos)*SVM.Params.alpha-SVM.Params.epsilon*sum(abs(SVM.Params.alpha))+sum(SVM.y(SVM.Params.pos)*SVM.Params.alpha); % Objective for optimal P with given alpha
elseif strcmp(SVM.type,'Classification')
    ObjNew = sum(SVM.Params.alpha)-.5*w'*K(SVM.Params.pos,SVM.Params.pos)*w; % Objective for optimal P with given alpha
end
SVM.Opt.dualGap2(end+1) = min(SVM.Opt.dualGap2(end),abs(SVM.Opt.Obj(end)-ObjNew));



%% Optimization line 4 ALG 2.1 
[SVM,Obj] = findAlpha_QP(SVM, Kernel, tau); % Update SVM with new kernel function
SVM.Opt.l(end+1)       = max(SVM.Opt.l(end),SVM.Opt.Obj(end) + sum(sum((P-Pold).*C)));
SVM.Opt.dualGap(end+1) = abs(Obj - SVM.Opt.l(end)); % Duality Gap
SVM.Opt.Obj(end+1) = Obj;
SVM.Opt.diff = (SVM.Opt.Obj(end)-SVM.Opt.Obj(end-1))./abs(SVM.Opt.Obj(end) + SVM.Opt.Obj(end-1))*200;
c = clock;
SVM.Opt.time(end+1) = c(end-2)*3600 + c(end-1)*60 + c(end);
SVM.Params.Cold = C;
end


