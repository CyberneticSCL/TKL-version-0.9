function [K] = gTKfast(x,y,a,b,delta,gamma,num)
%G Summary of this function goes here
%   Detailed explanation goes here
[n,dim] = size(x);
for j = 1:dim
    m(:,j) = max([x(:,j),y(:,j)]')';
end
q = size(delta,1);
for i = 1:2*q
    for j = 1:2*q
        K{i,j} = {zeros(num,num)};
    end
end

for i = 1:2*q
    for j = 1:2*q
        if (i <= q) & (j<=q)
            temp = prod(x.^(delta(i,:)).*y.^(delta(j,:)),2).*tTK(m,b,gamma(i,:)+gamma(j,:)+1);
            K{i,j} = reshape(temp,num,num)';
        elseif (i <= q) & (j>q)
            temp = prod(x.^(delta(i,:)).*y.^(delta(j-q,:)),2).*tTK(y,b,gamma(i,:)+gamma(j-q,:)+1);
            K{i,j} = reshape(temp,num,num)' - K{i,j-q};
        elseif (i > q) & (j<=q)
            temp =  prod(x.^(delta(i-q,:)).*y.^(delta(j,:)),2).*tTK(x,b,gamma(i-q,:)+gamma(j,:)+1);
            K{i,j} = reshape(temp,num,num)' - K{i-q,j};
        elseif (i > q) & (j > q)
            temp = prod(x.^(delta(i-q,:)).*y.^(delta(j-q,:)),2).*tTK(a,b,gamma(i-q,:)+gamma(j-q,:)+1);
            K{i,j} = reshape(temp,num,num)' - K{i,j-q} - K{i-q,j} - K{i-q,j-q};
        end
    end
end

end