function [SVM] = findP(SVM,Kernel)
%FINDP Updates the P matrix of the TK kernel
%   First input is trained SVM
%   Second input is Kernel object

%% Set up SDPT3 Variables
blk{1,1} = 's'; blk{1,2} = [SVM.Params.q];
cTemp = zeros(SVM.Params.q);
if strcmp(SVM.type,'Regression')
    w = SVM.Params.alpha;
elseif strcmp(SVM.type,'Classification')
    w = SVM.Params.alpha.*SVM.y(SVM.Params.pos)';
end

%% Calculate C
for i=1:length(Kernel.K)
    for j = 1:length(Kernel.K)
        cTemp(i,j) = -(w)'*Kernel.K{i,j}(SVM.Params.pos,SVM.Params.pos)*(w);
    end
end
C{1} = .5.*(cTemp+cTemp'); %Ensure C is symmetric.

%% Analytical Solution
[V,D] = eig(C{1});
V = V(:,find(min(diag(D)) == diag(D)));
P= length(V).*V*V';

Pold = SVM.Params.P;
eta = .5;
SVM.Params.P = (Pold + eta.*P)./(1+eta);
[SVM,Obj] = findAlpha(SVM,Kernel);
go = true;
while (Obj >= SVM.Opt.Obj(end)) & go
    eta = eta./10;
    if eta < .003
        go = false;
        SVM.Params.P = Pold;
        SVM.Opt.diff = 0;
    else
        SVM.Params.P = (Pold + eta.*P)./(1+eta);
        [SVM,Obj] = findAlpha(SVM,Kernel);
    end
end

if go
    SVM.Opt.Obj(end+1) = Obj;
    SVM.Opt.diff = abs(SVM.Opt.Obj(end)-SVM.Opt.Obj(end-1))./abs(SVM.Opt.Obj(end) + SVM.Opt.Obj(end-1))*200;
end

end

