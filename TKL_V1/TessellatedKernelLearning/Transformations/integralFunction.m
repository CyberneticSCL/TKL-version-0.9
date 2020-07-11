function [ SVM ] = integralFunction(SVM, exponent, bound )
%INTEGRALFUNCTION Summary of this function goes here
%   Detailed explanation goes here

xNew = SVM.xNew;
SVM.exponent = exponent;
SVM.bound = bound;
SVM.integral = 'Normal';

dim  = size(xNew,1);
NumD = size(xNew,2);


yBounds = [-(bound).*ones(dim,1), (1+bound).*ones(dim,1)];

%% Total Area
intDataT   = (1/(exponent))^dim*prod(yBounds(:,2).^exponent-yBounds(:,1).^exponent); % Total Area of Box
intDataU   = (1/(exponent))^dim*prod([kron(yBounds(:,2).^exponent,ones(1,NumD))-xNew.^exponent;ones(1,NumD)]);
intDataL   = intDataT - intDataU;

for n = 1:NumD
    for l = 1:dim
        xTemp(l,:) = max([xNew(l,:);kron(xNew(l,n),ones(1,NumD))]);
    end
    intDataUTemp(n,:) = (1/(exponent))^dim*prod([kron(yBounds(:,2).^exponent,ones(1,NumD))-xTemp.^exponent;ones(1,NumD)]);
    intDataULTemp(n,:)= kron(intDataU(n),ones(1,NumD)) - intDataUTemp(n,:);
    intDataLUTemp(n,:)= intDataU - intDataUTemp(n,:);
    intDataLTemp(n,:) = intDataT - intDataUTemp(n,:) - intDataULTemp(n,:) - intDataLUTemp(n,:);
end

I{1,1} = intDataUTemp;
I{1,2} = intDataULTemp;
I{2,1} = intDataLUTemp;
I{2,2} = 1/2.*(intDataLTemp+intDataLTemp');
SVM.Area{1} = max(I{1,1}(:)); SVM.Area{2} = max(I{2,2}(:)); SVM.Area{3} = intDataT;
SVM.I = I;
end