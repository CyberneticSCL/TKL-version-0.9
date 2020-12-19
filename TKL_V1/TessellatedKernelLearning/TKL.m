function [SVM] = TKL(x,y,type,C,params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM = TKL(x,y,type,C,params) function maps inputs x, to outputs
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
% NOTES:
% For support, contact M. Peet, Arizona State University at mpeet@asu.edu
% or B.K. Colbert at brendon.colbert@asu.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TKL - TKL
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
    params = paramsTKL();
elseif nargin == 4
    params = paramsTKL();
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
    SVM.xOld = x';
elseif length(y) == size(x,2)
    SVM.xOld = x;
else
    error('x and y must contain the same number of points')
end
SVM = scaling(SVM,SVM.xOld); % Scale the data to ensure numerical stability


%% Kernel Matrix set up
[dim,num] = size(SVM.x); % Dimension and number of inputs
SVM.Params.Lower = min(SVM.x')-params.bound; % Lower bounds of integration
SVM.Params.Upper = max(SVM.x')+params.bound; % Upper bounds of integration

%% Initialize Kernel Matrix
numK = num^2; yTemp = zeros(dim,numK); xTemp = yTemp; % Set up vectors of inputs
pos1 = 1; add = num-1;
for n = 1:num
    yTemp(:,pos1:pos1+add) = SVM.x;
    xTemp(:,pos1:pos1+add) = kron(SVM.x(:,n),ones(1,add+1));
    pos1 = pos1+add+1;
end
Kernel.K = initTK(xTemp',yTemp',SVM.Params.Lower,SVM.Params.Upper,num); % Initialize Kernel Matrix to speed up kernel matrix computation
Kernel.Z = monomials(SVM.x,params.deg); % Initialize matrix of monomials of our input data

%% Save Kernel Variables
SVM.Params.KernelType = 'TK'; SVM.Params.degree = params.deg; SVM.Params.q = 2.*size(Kernel.Z,1);

%% Set up TKL
q = SVM.Params.q; SVM.Params.P = eye(q); % Initialize P matrix
Eps = params.epsilon; tol = params.tol; maxit = params.maxit; SVM.Params.C = C; SVM.Params.epsilon = Eps; 
SVM.Opt.Obj = -inf; % Initialize Objective value

% Record SVM type
if (strcmp(type,'Classification') | strcmp(type,'classification') | strcmp(type,'C') | strcmp(type,'c'))
    SVM.type = 'Classification';    
else(strcmp(type,'Regression') | strcmp(type,'regression') | strcmp(type,'R') | strcmp(type,'r'))
    SVM.type = 'Regression';
end

%% TKL Algorithm
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
[SVM,SVM.Opt.Obj(end+1)] = findAlpha(SVM,Kernel); % Calculates alpha for final P matrix value.

