function [params] = paramsTK(degree,bound,epsilon,maxit,tol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [params] = paramsTK(degree,bound,epsilon,maxit,tol) function takes
% degree, bound, epsilon, maxit, and tol parameters as inputs for TK kernel
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% PMKL - paramsTK
%
%
% This program is provided to the reviewers of NeurIPS 2021.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Default Parameters
params.kernel = 'TK';
params.degree = 1; % Degree
params.bound = .5; % Bounds of iteration
params.epsilon = .1; % Epsilon
params.maxit   = 100; % Maximum iterations
params.tol     = 1e-2; % Stopping tolerance


%% Insert non-default parameters depending on the arguments supplied as inputs.
if nargin == 1
    if ~isempty(degree)
        params.degree = degree;
    end
elseif nargin == 2
    if ~isempty(degree)
        params.degree = degree;
    end
    if ~isempty(bound)
        params.bound = bound;
    end
elseif nargin == 3
    if ~isempty(degree)
        params.degree = degree;
    end
    if ~isempty(bound)
        params.bound = bound;
    end
    if ~isempty(epsilon)
        params.epsilon = epsilon;
    end
elseif nargin == 4
    if ~isempty(degree)
        params.degree = degree;
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
        params.degree = degree;
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
params.regul_alpha  = 0;
params.restart_iter = 10000;
end

