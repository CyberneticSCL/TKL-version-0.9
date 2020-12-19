function [SVM,Obj] = findAlpha(SVM,Kernel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [SVM,Obj] = findAlpha(SVM,Kernel) function takes a support vector
% machine object and a kernel object and optimizes the support vector
% machine with respect to the current kernel function.
% 
% INPUT
% SVM:    Optimized TK SVM (output of TKL function).
% Kernel: An internal kernel object used for quickly calculating the kernel matrix.
%
% OUTPUT
% SVM:    Optimized TK SVM (for current TK kernel).
% Obj:    The objective value of the SVM with the current TK kernel.
% 
% NOTES:
% For support, contact M. Peet, Arizona State University at mpeet@asu.edu
% or B.K. Colbert at brendon.colbert@asu.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TKL - findAlpha
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

