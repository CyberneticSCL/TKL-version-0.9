function [ X ] = monomialSymFunction( x, d )
%MONOMIALFUNCTION Summary of this function goes here
%   Detailed explanation goes here

Dim = size(x,1);
index = [];

if d > 0
    Vec = pvar('p');
    for n = 1:Dim
        VecTemp{n} = x(n,:);
        Vec(n,:) = x(n,:);
        index(end+1) = n;
    end
end

for k = 1:d-1;
    VecNew = {};
    indexNew = [];
    for n = 1:length(VecTemp)
        for l = index(n):Dim
            VecNew{end+1}   = VecTemp{n}.*x(l,:);
            Vec(end+1,:)    = VecNew{end};
            indexNew(end+1) = l;
        end
    end
    VecTemp = VecNew;
    index   = indexNew;
end
X = [ones(1,size(x,2));Vec];
end