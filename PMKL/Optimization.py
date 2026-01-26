import numpy as np
import itertools
import math
from matplotlib import pyplot as plt
from libsvm import svmutil
from PMKL import KernelFunctions
makeK  = KernelFunctions.makeK

def findAlpha(SVM, Kernel):
    '''
    SVM, Obj = findP(SVM,Kernel) function takes a support vector machine,
    object and a kernel object and finds an update to the P matrix of the
    kernel function.
    
    INPUT
    SVM:    SVM object.
    Kernel: An internal kernel object used for quickly calculating the kernel matrix.
    OUTPUT
    SVM:    Optimized SVM (for given kernel function).
    '''
    if SVM.Type == 'Classification':
        SVM.Params.K = makeK(Kernel, SVM.Params.P) # Make kernel matrix

        K1 = np.concatenate([np.arange(1, len(SVM.y)+1)[:, np.newaxis], SVM.Params.K], axis = 1) # include sample serial number as first column
#         print(K1.shape)
#         K1 = SVM.Params.K # include sample serial number as first column
#         SVM.model.fit(K1, np.ravel(SVM.y))
        model = svmutil.svm_train(np.ravel(SVM.y), K1, '-t 4 -s 0 -c ' + str(SVM.Params.C) ) # Use LibSVM to optimize SVM
#         print(model.get_sv_coef())
        sv_coef = np.array(model.get_sv_coef())
        SVs = np.array(model.get_sv_indices(), dtype = np.int32)
#         print(sv_coef.shape, SVs.shape)
#         sv_coef = SVM.model.dual_coef_
#         SVs = SVM.model.support_
#         rho = SVM.model.intercept_
        SVM.SVM_MODEL = model;
#         print(sv_coef.shape, SVs.shape)
#         Obj = np.sum(np.abs(sv_coef))-0.5*sv_coef@SVM.Params.K[SVs,SVs]@sv_coef # The objective value of the SVM
        SVM.Params.pos = SVs-1
        SVM.Params.alpha = np.abs(sv_coef) # The position of the support vectors and their values
#         alpha_dec = np.zeros(len(SVM.Params.K))
#         alpha_dec[SVM.Params.pos] = SVM.Params.alpha[:, 0]
#         print(alpha_dec)
#         SVM.Params.b = -rho # The b parameter of the SVM
        Obj = -0.5*sv_coef.T@SVM.Params.K[ SVM.Params.pos, :][:, SVM.Params.pos]@sv_coef  + np.sum(np.abs(SVM.Params.alpha))
        SVM.Params.sv_coef = sv_coef
        SVM.Params.rho = np.mean( SVM.Params.K[:, SVM.Params.pos]@sv_coef - SVM.y)
    elif SVM.Type == 'Regression' :
        SVM.Params.K = makeK(Kernel, SVM.Params.P) # Make kernel matrix
#         K1 = SVM.Params.K  

        K1 = np.concatenate([np.arange(1, len(SVM.y)+1)[:, np.newaxis], SVM.Params.K], axis = 1)# include sample serial number as first column
        model = svmutil.svm_train(np.ravel(SVM.y), K1, '-t 4 -s 3 -e ' + str(SVM.Params.epsilon) + ' -c ' + str(SVM.Params.C))
#         SVM.model.fit(K1, np.ravel(SVM.y))

        SVM.SVM_MODEL = model;
        sv_coef = np.array(model.get_sv_coef())
        SVs = np.array(model.get_sv_indices(), dtype = np.int32)
#         rho = SVM.model.intercept_
        
#         print(SVM.Params.K)
        SVM.Params.pos =  SVs-1
        SVM.Params.alpha = sv_coef # The position of the support vectors and their values
#         SVM.Params.b = -rho # The b parameter of the SVM
#         print(sv_coef.shape, SVs.shape)
#         print(SVM.y.shape, SVM.Params.pos.shape)
        Obj = -0.5*sv_coef.T@SVM.Params.K[ SVM.Params.pos, :][:, SVM.Params.pos]@sv_coef -SVM.Params.epsilon*np.sum(np.abs(SVM.Params.alpha)) + np.sum(SVM.y[SVM.Params.pos]*sv_coef) # The objective value of the SVM
        SVM.Params.rho = np.mean( SVM.Params.K[:, SVM.Params.pos]@sv_coef - SVM.y)
    return np.array(Obj).item()



def findP(SVM, Kernel):
    '''
    [SVM,Obj] = findP(SVM,Kernel) function takes a support vector machine,
    object and a kernel object and finds an update to the P matrix of the
    kernel function.
    
    INPUT
    SVM:    SVM object.
    Kernel: An internal kernel object used for quickly calculating the kernel matrix.
    OUTPUT
    SVM:    Optimized SVM (for given kernel function).
    '''
    cTemp = np.zeros((SVM.Params.q, SVM.Params.q));
    if SVM.Type == 'Regression':
        w = SVM.Params.alpha # w depends on the type of SVM
    elif SVM.Type == 'Classification':
        w = SVM.Params.alpha*SVM.y[SVM.Params.pos]# w depends on the type of SVM
    subset_Z = Kernel.subset_Z
    for i in range(1, subset_Z*Kernel.Z.shape[1] + 1):
        for j in range(1, subset_Z*Kernel.Z.shape[1] + 1):
            n = (i > Kernel.Z.shape[1]) + 1
            m = (j > Kernel.Z.shape[1]) + 1
            
            k = i - (n-1)*Kernel.Z.shape[1]
            l = j - (m-1)*Kernel.Z.shape[1]
#             print((Kernel.Z[:, k-1][SVM.Params.pos][:, np.newaxis]@Kernel.Z[:, l-1 ][ SVM.Params.pos][:, np.newaxis].T).shape)
#             print(n,m,k,l, len(SVM.Params.pos), Kernel.Z[:, k-1][SVM.Params.pos].T@Kernel.Z[:, l-1 ][ SVM.Params.pos])
            kTemp = Kernel.K[n,m][ SVM.Params.pos, :][:, SVM.Params.pos] *(Kernel.Z[:, k-1][SVM.Params.pos][:, np.newaxis]@Kernel.Z[:, l-1 ][ SVM.Params.pos][:, np.newaxis].T)
#             print(kTemp.shape, w.shape)
            cTemp[i-1, j-1]= -0.5*w.T@kTemp@w;
#     print(cTemp[0:2, 0:2])
    C = 0.5*(cTemp + cTemp.T)
#     print(cTemp[:5, :5])
    D, V  = np.linalg.eig(C)#  Calculate eigenvalues and eigenvectors
    V = V[:, np.argmin(D)]#  Select the eigenvector that corresponds to the minimum eigenvalue
#     print(V)
    P= len(V)*V[:,np.newaxis]@V[:,np.newaxis].T#  Calculate optimal P matrix
    P = np.real(P)
#     print(P[0:2, 0:2])
    ### Update P
    Pold = SVM.Params.P # Previous P matrix
#     print(Pold[0:2, 0:2])
#     print(np.linalg.norm(P-Pold))
    if len(SVM.Opt.StepLength) == 1:
        eta = 1;
    else:
        eta = np.mean(SVM.Opt.StepLength) # Step length
    etaMin = eta*1e-3;

    
    ########### Calculate Dual Gap
#     print(P[:5, :5])
    K = makeK( Kernel,P) # Make kernel matrix
    
    if  SVM.Type =='Regression':
        ObjNew = -.5*w.T@K[:, SVM.Params.pos][SVM.Params.pos, :]@w-SVM.Params.epsilon*np.sum(np.abs(SVM.Params.alpha))+np.sum(SVM.y[SVM.Params.pos]*SVM.Params.alpha)  # Objective for optimal P with given alpha
    elif  SVM.Type == 'Classification':
        ObjNew = np.sum(SVM.Params.alpha)-0.5*w.T@K[:, SVM.Params.pos][SVM.Params.pos, :]@w # Objective for optimal P with given alpha

    ObjNew = np.array(ObjNew).item()
#     print(ObjNew)
    SVM.Opt.dualGap2.append( np.min([SVM.Opt.dualGap2[-1],SVM.Opt.Obj[-1]-ObjNew]))
    
    SVM.Params.P = Pold + eta*(P-Pold);
    Obj  = findAlpha(SVM,Kernel) # Update alpha
#     print(eta, Obj)
    go = True
    stepDecrease = 5
    while (Obj >= SVM.Opt.Obj[-1]) & go:
        eta = eta/stepDecrease # Decrease step length
        if eta <= etaMin: # Minimum step length
            eta = etaMin
            go  = False # ends iteration
            SVM.Params.P = Pold + eta*(P-Pold) 
            Obj = findAlpha(SVM,Kernel) # Update alpha
        else:
            SVM.Params.P = Pold + eta*(P-Pold) 
            Obj = findAlpha(SVM,Kernel) # Update alpha

#         print(eta, Obj)
                            
    SVM.Opt.l.append(np.max([SVM.Opt.l[-1], SVM.Opt.Obj[-1] + np.sum(np.sum((P-Pold)*C))]))
    SVM.Opt.dualGap.append(np.abs(Obj - SVM.Opt.l[-1])) # Duality Gap


    SVM.Opt.Obj.append( Obj) # Update objective value
    SVM.Opt.diff = np.abs(SVM.Opt.Obj[-1]-SVM.Opt.Obj[-2]) /np.abs(SVM.Opt.Obj[-1] + SVM.Opt.Obj[-2])*200 # Update the percentage difference in the Objective function
    SVM.Opt.StepLength.append( eta)      