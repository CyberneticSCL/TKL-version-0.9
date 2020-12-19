function [params] = paramsTKL(degree,bound,epsilon,maxit,tol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [params] = paramsTKL(degree,bound,epsilon,maxit,tol) function takes
% degree, bound, epsilon, maxit, and tol parameters as inputs for TKL
% functions.  Entering an empty array as an input means the default value
% will be used.
% 
% INPUT
% degree:  Degree of the TK Kernel function (delta parameter, gamma = 0)
% bound:   Area of integration is [-bound,1+bound]
% epsilon: Epsilon parameter for regression problems
% maxit:   Maximum number of iterations.
% tol:     Tolerance of the TKL optimization algorithm.
%
% OUTPUT
% params: A variable containing the saved parameter values to be given to TKL.  
% 
% NOTES:
% For support, contact M. Peet, Arizona State University at mpeet@asu.edu
% or B.K. Colbert at brendon.colbert@asu.edu

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% TKL - paramsTKL
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

%% Default Parameters
params.deg = 1; % Degree
params.bound = .5; % Bounds of iteration
params.epsilon = .1; % Epsilon
params.maxit   = 100; % Maximum iterations
params.tol     = 1e-3; % degree


%% Insert non-default parameters depending on the arguments supplied as inputs.
if nargin == 1
    if ~isempty(degree)
        params.deg = degree;
    end
elseif nargin == 2
    if ~isempty(degree)
        params.deg = degree;
    end
    if ~isempty(bound)
        params.bound = bound;
    end
elseif nargin == 3
    if ~isempty(degree)
        params.deg = degree;
    end
    if ~isempty(bound)
        params.bound = bound;
    end
    if ~isempty(epsilon)
        params.epsilon = epsilon;
    end
elseif nargin == 4
    if ~isempty(degree)
        params.deg = degree;
    end
    if ~isempty(bound)
        params.bound = bound;
    end
    if ~isempty(epsilon)
        params.epsilon = epsilon;
    end
    if ~isempty(maxit)
        params.maxit = maxit;
    end
elseif nargin == 5
    if ~isempty(degree)
        params.deg = degree;
    end
    if ~isempty(bound)
        params.bound = bound;
    end
    if ~isempty(epsilon)
        params.epsilon = epsilon;
    end
    if ~isempty(maxit)
        params.maxit = maxit;
    end
    if ~isempty(tol)
        params.tol = tol;
    end
end

end

