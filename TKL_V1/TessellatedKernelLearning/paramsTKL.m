function [params] = paramsTKL(degree,bound,epsilon,maxit,tol)
%PARAMSTKL Summary of this function goes here
%   Detailed explanation goes here
params.deg = 1;
params.bound = .5;
params.epsilon = .1;
params.maxit   = 100;
params.tol     = 1e-3;

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

