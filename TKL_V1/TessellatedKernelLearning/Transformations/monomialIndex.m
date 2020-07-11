function [In] = monomialIndex(dim, d)
%MONOMIALS Sets up monomials, can take symbolic or real data.
%%Inputs:
% x - Data to transform
%     - Columns are individual data points.
%     - Rows represent the dimension of the points.
% d - Degree of the monomials.
% zero - Include zero'th power if 1, do not include if 0.
%%Outputs:
% z - transformed x matrix.

In = zeros(1,dim);
index = [];
if d > 0
    for n = 1:dim
        In(n,n) = 1;
        index(end+1) = n;
    end
end
InTemp = In;
for k = 1:d-1
    InNew = [];
    indexNew = [];
    for n = 1:size(InTemp,1)
        for l = index(n):dim
            InNew(end+1,:) = InTemp(n,:);
            InNew(end,l) = InNew(end,l)+1;
            indexNew(end+1) = l;
        end
    end
    In = [In;InNew];
    InTemp = InNew;
    index = indexNew;
end
In = [zeros(1,dim);In];
end