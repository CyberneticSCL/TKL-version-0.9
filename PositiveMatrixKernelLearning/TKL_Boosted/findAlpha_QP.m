function [SVM,Obj] = findAlpha_QP(SVM, Kernel, tau )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [SVM,Obj] = findAlpha_QP(SVM,Kernel) function takes a support vector
% machine object and a kernel object and optimizes the support vector
% machine with respect to the current kernel function.
% 
% INPUT
% SVM:    SVM object.
% Kernel: An internal kernel object used for quickly calculating the kernel matrix.
% tau   : Parameter used for APD
%
% OUTPUT
% SVM:    SVM object.
% Obj:    The objective value of the SVM with the current kernel.
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - findAlpha_QP
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Default Parameters 
tol = SVM.Params.tol/10; % Epsilon
val = +inf;
if nargin == 2
    tau = val;
end 
%% Insert non-default parameters depending on the arguments supplied as inputs.
 
if strcmp(SVM.type,'Classification')
    %%%%%%%%%%%%%%%%%% SOLUTION OF QP %%%%%%%%%%%%%%%%%%%%
    alpha_p = zeros(size(SVM.y'));
    alpha_p(SVM.Params.pos) = SVM.Params.alpha ;
    [SVM.Params.K] = makeK(SVM,Kernel); % Make kernel matrix
    K = SVM.Params.K; 
    Q = tau*diag(SVM.y)* K*diag(SVM.y) + eye(size(K)) + tau*SVM.Params.regul_alpha*eye(size(K)); %% Add regulaziation term
 
    Q1 = diag(SVM.y)*K*diag(SVM.y)  + SVM.Params.regul_alpha*eye(size(K))  ;

    c =  -tau*ones(size(alpha_p))- alpha_p  ;
 
    a = SVM.y;
    
    bux = SVM.Params.C*ones(size(a));
    blx = zeros(size(a));
    %% QP PROBLEM
    options = optimoptions(@quadprog, 'Algorithm', 'interior-point-convex', 'Display', 'off', 'MaxIterations', 10000, 'ConstraintTolerance', tol, 'OptimalityTolerance',tol, 'StepTolerance', tol );

    [x,fval,exitflag,output,lambda] = quadprog(Q, c, [], [], a, [0], blx, bux, alpha_p, options);
%     [x, lambda, pos, mu] = monqp(Q, -c, SVM.y', [0], SVM.Params.C, SVM.Params.regul_alpha,0, alpha_p);
    new_alpha = x;
    pos = abs(new_alpha) > SVM.Params.tol;
    pos = find(pos == 1);
    alpha = new_alpha(pos);  
    %% COMPUTE OBJ
    Obj = sum(abs(alpha))-.5*alpha'*Q1(pos,pos) *alpha;
    
    SVM.Params.alpha = alpha;
    SVM.Params.pos   =  pos; 
    svm_alpha = zeros(size(alpha_p));
    svm_alpha(pos) = abs(alpha);
    SVM.Opt.diff_alpha = norm(svm_alpha - alpha_p);
    
    
elseif strcmp(SVM.type,'Regression')
    %%%%%%%%%%%%%%%%%% LIBSVM %%%%%%%%%%%%%%%%%%%%
    alpha_p = zeros(size(SVM.y));
    alpha_p(SVM.Params.pos) = SVM.Params.alpha;
    [SVM.Params.K] = makeK (SVM,Kernel); % Make kernel matrix
%     [SVM.Params.K] = makeK_2(SVM,Kernel); % Make kernel matrix
    Q = SVM.Params.K + SVM.Params.regul_alpha*eye(size(SVM.Params.K)) + 1/tau*eye(size(SVM.Params.K));% Add regulaziation term
%     Q = (Q+Q')/2;
    Qp = [Q -Q; -Q Q];
%     Q = 1/tau*eye(size(SVM.Params.K));
    Q1 = SVM.Params.K  + SVM.Params.regul_alpha*eye(size(SVM.Params.K));

    cu =  SVM.Params.epsilon*ones(size(alpha_p)) + SVM.y + (1/tau)*alpha_p   ;
    cl =  SVM.Params.epsilon*ones(size(alpha_p)) - SVM.y - (1/tau)*alpha_p   ;
    c = [cu, cl];
%     c1 = -ones(size(alpha_p));
    a = [ones(size(alpha_p)), -ones(size(alpha_p))];
    
    bux = SVM.Params.C*ones(size(a));
    blx = zeros(size(a)); 
%     blc = 0;
%     buc = 0;
%     options = optimoptions('quadprog',  'Display', 'off', 'MaxIterations', 1000, 'ConstraintTolerance', tol, 'OptimalityTolerance',tol, 'StepTolerance', tol );
    %% QP PROBLEM
    [x,fval,exitflag,output,lambda] = quadprog(Qp, c, [], [], a, [0], blx, bux, 0);

    new_alpha = -x(1:size(alpha_p, 2)) + x(size(alpha_p, 2)+1:end);
    pos = abs(new_alpha) > SVM.Params.tol;
    pos = find(pos == 1);
    alpha = new_alpha(pos); 
    %% COMPUTE OBJ
    Obj = 0.5*alpha'*SVM.Params.K(pos,pos)*alpha + SVM.Params.epsilon*sum(abs(alpha)) - sum(SVM.y(pos)*alpha); % The objective value of the SVM
    Obj = - Obj;    
    SVM.Params.alpha = alpha;
    SVM.Params.pos   = pos;

    SVM.Opt.diff_alpha = norm(new_alpha - alpha_p');
else
    error('No type of SVM specified')
    end
end

