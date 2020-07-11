function [Kt] = makeTKTest(SVM,xTest)
%%
xtrain = SVM.x(:,SVM.Params.pos);
[dimx,numx] = size(xtrain); numtest = size(xTest,2);

y = zeros(dimx,numx*numtest); x = y;
pos1 = 1; add = numx-1;
for n = 1:numtest
    y(:,pos1:pos1+add) = kron(xTest(:,n),ones(1,numx));
    x(:,pos1:pos1+add) = xtrain;
    pos1 = pos1+add+1;
end

Kt = gTKtest(x',y',SVM.Params.Lower,SVM.Params.Upper,SVM.Params.delta,SVM.Params.gamma,numx,numtest,SVM.Params.P)';
end

