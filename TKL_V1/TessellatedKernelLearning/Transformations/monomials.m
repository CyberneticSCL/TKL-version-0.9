function [z,univariateAll,variateAll] = monomials(x, d, zero)
%MONOMIALS Sets up monomials, can take symbolic or real data.
%%Inputs:
% x - Data to transform
%     - Columns are individual data points.
%     - Rows represent the dimension of the points.
% d - Degree of the monomials.
% zero - Include zero'th power if 1, do not include if 0.
%%Outputs:
% z - transformed x matrix.

if nargin < 3
    zero = 1;
end

dim = size(x,1);
index = [];
univariate = [];

% Allow for symbolic variables to be passed.
if strcmp(class(x),'sym')
    vec = sym(zeros(dim,1));
else
    vec = [];
end

if d > 0
    for n = 1:dim
        vecTemp{n} = x(n,:);
        vec(n,:) = x(n,:);
        index(end+1) = n;
        univariate(end+1) = n;
        variate(n,n)      = 1;
    end
end
univariateAll = univariate; variateAll = variate; variateTemp = variate;
for k = 1:d-1
    vecNew = {};
    indexNew = [];
    univariateNew = [];
    variateNew    = [];
    for n = 1:length(vecTemp)
        for l = index(n):dim
            vecNew{end+1} = vecTemp{n}.*x(l,:);
            vec(end+1,:)  = vecNew{end};
            indexNew(end+1) = l;
            tempvar = zeros(1,dim); tempvar(l) = 1;
            variateNew(:,end+1) = variateTemp(:,n)+tempvar';
            if univariate(n) == l
                univariateNew(end+1) = l;
            else
                univariateNew(end+1) = 0;
            end
        end
    end
    univariateAll = [univariateAll,univariateNew];
    univariate    = univariateNew;
    variateAll    = [variateAll,variateNew];
    variateTemp   = variateNew;
    vecTemp = vecNew;
    index = indexNew;
end
if zero == 1;
    z = [ones(1,size(x,2));vec];
else
    z = vec;
end
end

