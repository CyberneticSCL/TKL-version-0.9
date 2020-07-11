function [ SVM ] = integralFunctionTest(SVM )
%INTEGRALFUNCTION Summary of this function goes here
%   Detailed explanation goes here

exponent = SVM.exponent;
bound    = SVM.bound;

xNew = SVM.xNew;
xTest= SVM.xTest;


dim      = size(xNew,1);
NumD     = size(xNew,2);
NumDt = size(xTest,2);

yBounds = [-(bound).*ones(dim,1), (1+bound).*ones(dim,1)];

%% Total Area
intDataT     = (1/(exponent))^dim*prod(yBounds(:,2).^exponent-yBounds(:,1).^exponent); % Total Area of Box
intDataU     = (1/(exponent))^dim*prod([kron(yBounds(:,2).^exponent,ones(1,NumD))-xNew.^exponent;ones(1,NumD)]);
intDataUTest = (1/(exponent))^dim*prod([kron(yBounds(:,2).^exponent,ones(1,NumDt))-xTest.^exponent;ones(1,NumDt)]);
intDataL     = intDataT - intDataU;

for n = 1:NumD
    for l = 1:dim
        xTemp(l,:) = max([xTest(l,:);kron(xNew(l,n),ones(1,NumDt))]);
    end
    intDataUTemp(n,:) = (1/(exponent))^dim*prod([kron(yBounds(:,2).^exponent,ones(1,NumDt))-xTemp.^exponent;ones(1,NumDt)]);
    intDataULTemp(n,:)= kron(intDataU(n),ones(1,NumDt)) - intDataUTemp(n,:);
    intDataLUTemp(n,:)= intDataUTest - intDataUTemp(n,:);
    intDataLTemp(n,:) = intDataT - intDataUTemp(n,:) - intDataULTemp(n,:) - intDataLUTemp(n,:);
end

I{1,1} = intDataUTemp;
I{1,2} = intDataULTemp;
I{2,1} = intDataLUTemp;
I{2,2} = intDataLTemp;
SVM.ITest = I;
end