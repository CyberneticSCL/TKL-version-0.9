import numpy as np
import itertools
import math
from matplotlib import pyplot as plt
from libsvm import svmutil
from Fast_TKL import KernelFunctions
from Fast_TKL import low_rank_QP
from Fast_TKL import Low_Rank_Kernel_Decomp

import scipy as sp
from scipy.sparse.linalg import eigsh
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import LinearOperator, eigsh

makeK  = KernelFunctions.makeK



def findAlpha_lowRankQP(SVM, Kernel, tol =1.e-5, tau0 = 1.e-5): #Y, eigvals_of_Gij, vectors_of_Gij, monomials_V, index_Z, index_G, P, rank, tau0 = 1.e-5):
    
    if SVM.Type == 'Classification':
        Y = SVM.y[:, 0]
        eigvals_of_Gij = Kernel.eigs
        vectors_of_Gij = Kernel.vecs
        monomials_V  = Kernel.Z
        index_Z = Kernel.index_Z
        index_G = Kernel.index_G
        P    = SVM.Params.P
        rank = SVM.rank
        additional_rank = SVM.additional_rank

    #     print(eigvals_of_Gij.shape, vectors_of_Gij.shape, monomials_V.shape, index_Z.shape, index_G.shape, P.shape)
        Kernel_vector_product = lambda vector: KernelFunctions.fast_full_kernel_vector_v2(vector, 
                                        eigvals_of_Gij, vectors_of_Gij, monomials_V, index_Z, index_G, P, SVM.Params.add_poly)

        # E, V = np.linalg.eigh(P)
        # E = np.sqrt(np.abs(E)) # added abs(E) for numerical stability. For SPD E can be -1.e-16
        # L = V*E
        # monomials_VP = monomials_V@L
        
        # Kernel_vector_product = lambda vector: KernelFunctions.fast_full_kernel_vector_v3(vector, 
        #                                 eigvals_of_Gij, vectors_of_Gij, monomials_V, monomials_VP, SVM.Params.add_poly)
        #     print(np.prod(grid_structure))

        N_samples = len(Y)
        A = LinearOperator((N_samples,N_samples), matvec=Kernel_vector_product) # linear operator for EigDec
        lambdas, vectors = eigsh(A, k=rank)

        ind_sorted = np.argsort(-lambdas)
        SVM.Params.eigvec = vectors
        SVM.Params.eigval = lambdas
        
        V_Q = vectors*np.sqrt(lambdas) # features of kernel
        D_Q = tau0*np.ones(N_samples) # 

        VVQ = (V_Q.T*Y).T
        # VVQ = VVQ/np.sqrt(lambdas.max())
#         SVM.Kernel.KER_VEC = vectors
#         SVM.Kernel.KER_lam = lambdas
        u = SVM.Params.C
        l = 0

        c = np.ones((N_samples, 1))
        b = Y[:, np.newaxis]


        alpha0 = np.ones((N_samples, 1))*(u + l)/2
        mu10   = np.ones((N_samples, 1))
        mu20   = np.ones((N_samples, 1))
        y0     = np.array([[1]])

        dec_var = low_rank_QP.solve_QP_IPM(D_Q, VVQ, c, b, l, u, alpha0, y0, mu10, mu20, 
                                       max_iter = 200, tol = 1.e-6, print_losses=0) 
        alpha_dec = dec_var[0]
#         print(alpha_dec)
        # VVQ = VVQ*np.sqrt(lambdas.max())
        r = (VVQ.T@alpha_dec)
        SVM.Params.alpha = alpha_dec
        rr = (V_Q.T@alpha_dec)
        SVM.Params.rho = np.mean( V_Q@r - SVM.y)
        
        Obj = -0.5*r.T@r + c.T@alpha_dec - 0.5*tau0*alpha_dec.T@alpha_dec
        
    elif SVM.Type == 'Regression' :
        # Find Eigen Decomposition of Kernel Matrix
        Y = SVM.y[:, 0]
        # Load precomputed decomposition of G_{ij}
        eigvals_of_Gij = Kernel.eigs
        vectors_of_Gij = Kernel.vecs
        monomials_V  = Kernel.Z
        index_Z = Kernel.index_Z
        index_G = Kernel.index_G
        P    = SVM.Params.P
        rank = SVM.rank
        additional_rank = SVM.additional_rank

        # define Kernel Matrix To Vector Product
        #     print(eigvals_of_Gij.shape, vectors_of_Gij.shape, monomials_V.shape, index_Z.shape, index_G.shape, P.shape)
        Kernel_vector_product = lambda vector: KernelFunctions.fast_full_kernel_vector_v2(vector, 
                                eigvals_of_Gij, vectors_of_Gij, monomials_V, index_Z, index_G, P, SVM.Params.add_poly)
        #     print(np.prod(grid_structure))
        E, V = np.linalg.eigh(P)
        E = np.sqrt(np.abs(E))
        L = V*E
        monomials_VP = monomials_V@L
        
        # Kernel_vector_product = lambda vector: KernelFunctions.fast_full_kernel_vector_v3(vector, 
                                        # eigvals_of_Gij, vectors_of_Gij, monomials_V, monomials_VP, SVM.Params.add_poly)
        # parameters of SVM
        N_samples = len(Y)
        vareps = SVM.Params.epsilon
        
        # Eigen Decomposition of Kernel Matrix
        A = LinearOperator((N_samples,N_samples), matvec=Kernel_vector_product) # linear operator for EigDec
        lambdas, vectors = eigsh(A, k=rank)

        
        ind_sorted = np.argsort(-lambdas)
        SVM.Params.eigvec = vectors
        SVM.Params.eigval = lambdas
        
        
#         print(A.shape, rank)
        # features for linear SVR
        V_Q = vectors*np.sqrt(np.abs(lambdas)) # features of kernel
        D_Q = tau0*np.ones(N_samples) #  

        V_Q = np.nan_to_num(V_Q)
        D_Q = np.nan_to_num(D_Q)

#         print(vectors, lambdas)
        
        # construct for low rank QP
        DD_Q= tau0*np.ones(2*N_samples)
        VVQ = np.block([[V_Q], [-V_Q]])
        # VVQ = VVQ/np.sqrt(lambdas.max())
        c =  np.block([[-vareps + Y[:, np.newaxis]], [-vareps - Y[:, np.newaxis]]]) 
        b =  np.ones((2*N_samples, 1))
        b[N_samples:,:] = -1 
#         SVM.Params.V_Q = V_Q
        u = SVM.Params.C
        l = 0
        
        # initial conditions
        alpha0 = np.ones((2*N_samples, 1))*(u + l)/2
        mu10   = np.ones((2*N_samples, 1))
        mu20   = np.ones((2*N_samples, 1))
        y0     = np.array([[1]])
        # low rank QP optimization
        dec_var = low_rank_QP.solve_QP_IPM(DD_Q, VVQ, c, b, l, u, alpha0, y0, mu10, mu20,
                       max_iter = 200, tol = 1.e-6, print_losses=0) 
        primal_loss = c.T@dec_var[0]
#         print(dec_var[-2], dec_var[-1], primal_loss, V_Q.shape)
        # dec variables
        alpha_dec = dec_var[0][:N_samples, :] - dec_var[0][N_samples:, :]
        # loss function
        r = (V_Q.T@alpha_dec)
        SVM.Params.alpha = alpha_dec
        SVM.Params.rho = np.mean( V_Q@r - SVM.y)
#         print(V_Q@V_Q.T)
        Obj = -0.5*r.T@r - 0.5*tau0*alpha_dec.T@alpha_dec - vareps*np.abs(alpha_dec).sum() + Y[np.newaxis, :]@alpha_dec
        
        
        
    else: 
        raise ValueError('SVM.Type is not specified! Set SVM.Type = "Classification" or "Regression"')
    return Obj[0, 0]



def findP_lowRank(SVM, Kernel):
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
     
    eigvals_of_Gij = Kernel.eigs
    vectors_of_Gij = Kernel.vecs
    monomials_V  = Kernel.Z
    index_Z = Kernel.index_Z
    index_G = Kernel.index_G

#     cTemp = np.zeros((SVM.Params.q, SVM.Params.q));
    if SVM.Type == 'Regression':
        w = SVM.Params.alpha # w depends on the type of SVM
    elif SVM.Type == 'Classification':
        w = SVM.Params.alpha*SVM.y# w depends on the type of SVM

    
    C = -KernelFunctions.compute_D_matrix(w[:, 0], eigvals_of_Gij, vectors_of_Gij, monomials_V, index_Z, index_G)
    
    
    D, V  = np.linalg.eig(C)#  Calculate eigenvalues and eigenvectors
    V = V[:, np.argmin(D)]#  Select the eigenvector that corresponds to the minimum eigenvalue
#     print(V)
    P= 2*len(V)*V[:,np.newaxis]@V[:,np.newaxis].T#  Calculate optimal P matrix
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

    
    ########### Calculate Dual Gap ############# 
    if  SVM.Type =='Regression':
        # primal Obj
        Obj1 = w.T@KernelFunctions.fast_full_kernel_vector_v2(w[:, 0], eigvals_of_Gij, vectors_of_Gij, 
                                                                       monomials_V, index_Z,
                                                                       index_G, P)
        # Dual Obj
        Obj2 = (-C*P).sum()
        if np.abs(Obj1-Obj2) > 1.e-5:
            print('ERROR')
            
            
        ObjNew =(-SVM.y.T@w + SVM.Params.epsilon*np.sum(np.abs(w))-0.5*w.T@KernelFunctions.fast_full_kernel_vector_v2(w[:, 0],
                                                            eigvals_of_Gij, vectors_of_Gij, 
                                                            monomials_V, index_Z,
                                                            index_G, P)) # Objective for optimal P with given alpha

    elif  SVM.Type == 'Classification':
        # primal Obj
        Obj1 = w.T@KernelFunctions.fast_full_kernel_vector_v2(w[:, 0], eigvals_of_Gij, vectors_of_Gij, 
                                                                       monomials_V, index_Z,
                                                                       index_G, P)
        # Dual Obj
        Obj2 = (-C*P).sum()
        if np.abs(Obj1-Obj2) > 1.e-5:
            print('ERROR')
        ObjNew =( np.sum(w)-0.5*w.T@KernelFunctions.fast_full_kernel_vector_v2(w[:, 0],
                                                                        eigvals_of_Gij, vectors_of_Gij, 
                                                                        monomials_V, index_Z,
                                                                        index_G, P)) # Objective for optimal P with given alpha

    ObjNew = np.max(ObjNew)
    SVM.Opt.dualGap2.append( np.min([SVM.Opt.dualGap2[-1],SVM.Opt.Obj[-1]-ObjNew]))
    
    SVM.Params.P = Pold + eta*(P-Pold);
    Obj  = findAlpha_lowRankQP(SVM,Kernel) # Update alpha
#     print(eta, Obj)
    go = True
    stepDecrease = 5
    while (Obj >= SVM.Opt.Obj[-1]) & go:
        eta = eta/stepDecrease # Decrease step length
        if eta <= etaMin: # Minimum step length
            eta = etaMin
            go  = False # ends iteration
            SVM.Params.P = Pold + eta*(P-Pold) 
            Obj = findAlpha_lowRankQP(SVM,Kernel) # Update alpha
        else:
            SVM.Params.P = Pold + eta*(P-Pold) 
            Obj = findAlpha_lowRankQP(SVM,Kernel) # Update alpha
            
#         print(eta, Obj) 
    SVM.Opt.l.append(np.max([SVM.Opt.l[-1], SVM.Opt.Obj[-1] + np.sum(np.sum((P-Pold)*C))]))
    SVM.Opt.dualGap.append(np.abs(Obj - SVM.Opt.l[-1])) # Duality Gap


    SVM.Opt.Obj.append( Obj) # Update objective value
    SVM.Opt.diff = np.abs(SVM.Opt.Obj[-1]-SVM.Opt.Obj[-2]) /np.abs(SVM.Opt.Obj[-1] + SVM.Opt.Obj[-2])*200 # Update the percentage difference in the Objective function
    SVM.Opt.StepLength.append( eta)      
    
    

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
        
#         print(sv_coef.shape, SVs.shape)
#         Obj = np.sum(np.abs(sv_coef))-0.5*sv_coef@SVM.Params.K[SVs,SVs]@sv_coef # The objective value of the SVM
        SVM.Params.pos = SVs-1
        SVM.Params.alpha = np.abs(sv_coef) # The position of the support vectors and their values
#         SVM.Params.b = rho # The b parameter of the SVM
        Obj = -0.5*sv_coef.T@SVM.Params.K[ SVM.Params.pos, :][:, SVM.Params.pos]@sv_coef  + np.sum(np.abs(SVM.Params.alpha))
    elif SVM.Type == 'Regression' :
        SVM.Params.K = makeK(Kernel, SVM.Params.P) # Make kernel matrix
#         K1 = SVM.Params.K  

        K1 = np.concatenate([np.arange(1, len(SVM.y)+1)[:, np.newaxis], SVM.Params.K], axis = 1)# include sample serial number as first column
        model = svmutil.svm_train(np.ravel(SVM.y), K1, '-t 4 -s 3 -e ' + str(SVM.Params.epsilon) + ' -c ' + str(SVM.Params.C))
#         SVM.model.fit(K1, np.ravel(SVM.y))

        sv_coef = np.array(model.get_sv_coef())
        SVs = np.array(model.get_sv_indices(), dtype = np.int32)
        
        SVM.Params.pos =  SVs-1
        SVM.Params.alpha = sv_coef # The position of the support vectors and their values
#         print(sv_coef.shape, SVs.shape)
#         print(SVM.y.shape, SVM.Params.pos.shape)
        Obj = -0.5*sv_coef.T@SVM.Params.K[ SVM.Params.pos, :][:, SVM.Params.pos]@sv_coef -SVM.Params.epsilon*np.sum(np.abs(SVM.Params.alpha)) + np.sum(SVM.y[SVM.Params.pos]*sv_coef) # The objective value of the SVM
    return Obj



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

    for i in range(1, 2*Kernel.Z.shape[1] + 1):
        for j in range(1, 2*Kernel.Z.shape[1] + 1):
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
    D, V  = np.linalg.eig(C)#  Calculate eigenvalues and eigenvectors
    V = V[:, np.argmin(D)]#  Select the eigenvector that corresponds to the minimum eigenvalue
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
    K = makeK( Kernel,P) # Make kernel matrix
    
    if  SVM.Type =='Regression':
        ObjNew = -.5*w.T@K[:, SVM.Params.pos][SVM.Params.pos, :]@w-SVM.Params.epsilon*np.sum(np.abs(SVM.Params.alpha))+np.sum(SVM.y[SVM.Params.pos]*SVM.Params.alpha)  # Objective for optimal P with given alpha
    elif  SVM.Type == 'Classification':
        ObjNew = np.sum(SVM.Params.alpha)-0.5*w.T@K[:, SVM.Params.pos][SVM.Params.pos, :]@w # Objective for optimal P with given alpha
    
#     print(ObjNew)
    SVM.Opt.dualGap2.append( np.min([SVM.Opt.dualGap2[-1],SVM.Opt.Obj[-1]-ObjNew]))
    
    SVM.Params.P = Pold + eta*(P-Pold);
    Obj  = findAlpha(SVM,Kernel) # Update alpha
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
                            
                            
    SVM.Opt.l.append(np.max([SVM.Opt.l[-1], SVM.Opt.Obj[-1] + np.sum(np.sum((P-Pold)*C))]))
    SVM.Opt.dualGap.append(np.abs(Obj - SVM.Opt.l[-1])) # Duality Gap


    SVM.Opt.Obj.append( Obj) # Update objective value
    SVM.Opt.diff = np.abs(SVM.Opt.Obj[-1]-SVM.Opt.Obj[-2]) /np.abs(SVM.Opt.Obj[-1] + SVM.Opt.Obj[-2])*200 # Update the percentage difference in the Objective function
    SVM.Opt.StepLength.append( eta)      