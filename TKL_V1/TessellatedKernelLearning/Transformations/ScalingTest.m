function [ X ] = ScalingTest( X, SF )
%SCALINGTEST Summary of this function goes here
%   Detailed explanation goes here
Dim = size(X,1);

for n = 1:Dim
    X(n,:) = (X(n,:) - SF(n,1))./(SF(n,2) - SF(n,1));
end

end

