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
% SVM:    Optimized TK SVM (for current TK kernel).
% 
% NOTES:
% For support, contact M. Peet, Arizona State University at mpeet@asu.edu
% or B.K. Colbert at brendon.colbert@asu.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TKL - findP
%
% Copyright (C)2019  M. Peet, B.K. Colbert
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% If you modify this code, document all changes carefully and include date
% authorship, and a brief description of modifications
%
% Initial coding MMP, BKC  - 12_15_2020
%

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
        cTemp(i,j) = -(w)'*kTemp*(w);
    end
end
C{1} = .5.*(cTemp+cTemp'); % Ensure C is symmetric.

%% Analytical Solution
[V,D] = eig(C{1}); % Calculate eigenvalues and eigenvectors
V = V(:,find(min(diag(D)) == diag(D))); % Select the eigenvector that corresponds to the minimum eigenvalue
P= length(V).*V*V'; % Calculate optimal P matrix

Pold = SVM.Params.P; % Previous P matrix
eta = .5; % Step length
SVM.Params.P = (Pold + eta.*P)./(1+eta); % Updated P matrix to update new kernel function
[SVM,Obj] = findAlpha(SVM,Kernel); % Update SVM with new kernel function

%% Simple line search if Objective value is larger than before
go = true;
while (Obj >= SVM.Opt.Obj(end)) & go
    eta = eta./10; % Decrease step length
    if eta < .003 % Minimum step length
        go = false; % ends iteration
        SVM.Params.P = Pold; % do not change kernel function
        SVM.Opt.diff = 0; % Objective value did not change
    else
        SVM.Params.P = (Pold + eta.*P)./(1+eta); % Test new step length and update P
        [SVM,Obj] = findAlpha(SVM,Kernel); % Update objective value and SVM parameters
    end
end

%% If step length is accepted 
if go
    SVM.Opt.Obj(end+1) = Obj; % Update objective value
    SVM.Opt.diff = abs(SVM.Opt.Obj(end)-SVM.Opt.Obj(end-1))./abs(SVM.Opt.Obj(end) + SVM.Opt.Obj(end-1))*200; % Update the percentage difference in the Objective function
end

end

