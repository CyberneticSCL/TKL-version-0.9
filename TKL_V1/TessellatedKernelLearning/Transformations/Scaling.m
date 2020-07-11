function [SVM] = Scaling(SVM,X)
%SCALING Summary of this function goes here
%   Detailed explanation goes here
Dim = size(X,1);

for n = 1:Dim
    ScaleFactor(n,:) = [min(X(n,:)),max(X(n,:))];
    if diff(ScaleFactor(n,:)) == 0
        ScaleFactor(n,:) = [-abs(min(X(n,:))),abs(min(X(n,:)))]
    end
    X(n,:) = (X(n,:) - ScaleFactor(n,1))./(ScaleFactor(n,2) - ScaleFactor(n,1));
end

SVM.x = X;
SVM.ScaleFactor = ScaleFactor;

end

