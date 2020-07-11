function [K] = gTKtest(x,y,a,b,delta,gamma,numx,numtest,P)
%G Summary of this function goes here
%   Detailed explanation goes here
[n,dim] = size(x);
for j = 1:dim
    m(:,j) = max([x(:,j),y(:,j)]')';
end
q = size(delta,1);
K = zeros(numx,numtest);
for i = 1:q
    for j = 1:q
        temp1 = prod(x.^(delta(i,:)).*y.^(delta(j,:)),2).*tTK(m,b,gamma(i,:)+gamma(j,:)+1);
        temp2 = prod(x.^(delta(i,:)).*y.^(delta(j,:)),2).*tTK(y,b,gamma(i,:)+gamma(j,:)+1)-temp1;
        temp3 = prod(x.^(delta(i,:)).*y.^(delta(j,:)),2).*tTK(x,b,gamma(i,:)+gamma(j,:)+1)-temp1;
        temp4 = prod(x.^(delta(i,:)).*y.^(delta(j,:)),2).*tTK(a,b,gamma(i,:)+gamma(j,:)+1)-temp1-temp2-temp3;
        LT = zeros(numx,numtest);
        LT = reshape(P(i,j).*temp1+P(i,j+q).*temp2 + P(i+q,j).*temp3 + P(i+q,j+q).*temp4,numx,numtest);
        K = K + LT;
    end
end

end