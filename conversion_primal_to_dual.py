import numpy as np
import scipy as sp

def primal_to_dual_variables_SVR(X, Y, primal_var, intercept, C, vareps, tol = 1.e-4, maxiter = 10000, rtol = 1.e-8):

    primal_output1 =  Y - primal_var@X.T - intercept + vareps # -xi*
    primal_output2 = -Y + primal_var@X.T + intercept + vareps # -xi
    # constraints where alpha = 0
    ALPHA_NC = np.argwhere(  primal_output1 < -tol)[:, 1] # alpha  =0, alpha* = C
    ALPHA_PC = np.argwhere(  primal_output2 < -tol)[:, 1] # alpha^*=0, alpha  = C


    ALPHA_N0 = np.argwhere( primal_output1 > tol)[:, 1] # alpha  >= 0, alpha*=0
    ALPHA_P0 = np.argwhere( primal_output2 > tol)[:, 1] # alpha* >= 0, alpha =0

    ALPHA_0 = np.intersect1d(ALPHA_P0, ALPHA_N0)

    reconstructed_alpha = np.zeros(len(X))
    reconstructed_alpha[ALPHA_NC] =  -C
    reconstructed_alpha[ALPHA_PC] =   C
#     print(reconstructed_alpha[ALPHA_PC])
    # constraints where alpha = C 
    
    # others alpha \not in ALPHA_0, ALPHA_NC, ALPHA_PC
    ALPHA_other = np.setdiff1d(np.arange(len(X), dtype = np.int64), 
                                         np.union1d(ALPHA_0, np.union1d(ALPHA_PC, ALPHA_NC)))
     
    if len(ALPHA_other) > 0:
    #     reconstructed_alpha[ALPHA_0]  =  0

        Xp1 = np.concatenate((X[ALPHA_other, :], np.ones((len(ALPHA_other), 1))), axis = 1)


        RHS_ = primal_var - reconstructed_alpha[ALPHA_NC]@X[ALPHA_NC, :]  - reconstructed_alpha[ALPHA_PC]@X[ALPHA_PC, :]
        RHS_ = np.append(RHS_, -reconstructed_alpha[ALPHA_NC].sum()-reconstructed_alpha[ALPHA_PC].sum())
    #     Mat_1 = Y[ALPHA_other]*X[ALPHA_other, :].T
    #     Mat_2 = Y[ALPHA_other][np.newaxis, :]
    #     Mat_ = np.concatenate((Mat_1, Mat_2), axis = 0)
    #     print(ALPHA_other.shape, PHI.shape, RHS_.shape)
    #     RHS_ = Y[ALPHA_other]*(Xp1@RHS_)


        Mat_constr = np.concatenate((X[ALPHA_other, :], np.ones((len(ALPHA_other), 1))), axis = 1).T
    #     Mat_constr.shape
#         print(Mat_constr.shape, RHS_.shape, ALPHA_other.shape, primal_var.shape)
    #     print((Mat_constr.T@Mat_constr).shape, (Mat_constr.T@RHS_).shape)
    #     mv = lambda vec : CG_matvec_product2(vec, Y[ALPHA_other], Xp1)
    #     # Mat_constr1 = YY[set_alpha_nonboundary]*phi1[set_alpha_nonboundary, :].T
    #     # Mat_constr2 = YY[set_alpha_nonboundary][np.newaxis, :]
    #     # Mat_constr = np.concatenate((Mat_constr1, Mat_constr2), axis = 0)
    #     A = LinearOperator((len(ALPHA_other),len(ALPHA_other)), matvec=mv)
    #     sol, info = sp.sparse.linalg.cg(A, RHS_, maxiter=maxiter, rtol =rtol)

        BOUNDS_L = -C*np.ones_like(ALPHA_other)
        BOUNDS_R = C*np.ones_like(ALPHA_other)

        ALPHA_P_OTHER = np.argwhere(ALPHA_other == np.intersect1d(ALPHA_N0, ALPHA_other)[:, np.newaxis])[:, 1]
        ALPHA_N_OTHER = np.argwhere(ALPHA_other == np.intersect1d(ALPHA_P0, ALPHA_other)[:, np.newaxis])[:, 1]
        BOUNDS_L[ALPHA_P_OTHER] = 0
        BOUNDS_R[ALPHA_N_OTHER] = 0

        res = sp.optimize.lsq_linear(Mat_constr, RHS_, bounds=(BOUNDS_L, BOUNDS_R), lsq_solver = 'exact', tol = 0.01*tol) 

#         print(reconstructed_alpha[ALPHA_PC])
    #     print(sol.shape, ALPHA_other.shape), Mat_constr, RHS_, ALPHA_other, ALPHA_0, ALPHA_NC, ALPHA_PC, ALPHA_N0, ALPHA_P0
        reconstructed_alpha[ALPHA_other] = res.x
#     print(reconstructed_alpha@X - primal_var)
#     print(reconstructed_alpha.sum())

    warning = 0
    if np.abs(reconstructed_alpha@X - primal_var).max()>tol:
        print('Warning! KKT Conditions |w - sum alpha_i x_i| > tol', np.abs(reconstructed_alpha@X - primal_var).max())
        warning = 1
        
    if np.abs(reconstructed_alpha.sum())>tol:
        print('Warning! KKT Conditions |  sum alpha_i | > tol', np.abs(reconstructed_alpha.sum()))
        warning = 1
    reconstructed_alpha_n = reconstructed_alpha.copy()
    reconstructed_alpha_n[reconstructed_alpha_n > 0] = 0
    
    if np.abs((reconstructed_alpha - reconstructed_alpha_n)*np.clip(primal_output2, 0, 1000)).max() > tol:
        print('Warning! KKT Conditions |  |alpha*_i(...)| | > tol', 
             np.abs((reconstructed_alpha - reconstructed_alpha_n)*np.clip(primal_output2, 0, 1000)).max())
        warning = 1
        
        
    if np.abs(reconstructed_alpha_n*np.clip(primal_output1, 0, 1000)).max()>tol:
        print('Warning! KKT Conditions |  |sum alpha_i(...) | > tol', 
              np.abs(reconstructed_alpha_n*np.clip(primal_output1, 0, 1000)).max())
        warning = 1
        
    if warning:
        print('Increase or decrease tolerance')
#     print(np.abs((reconstructed_alpha - reconstructed_alpha_n)*np.clip(primal_output2, 0, 1000)).max())
#     print(np.abs(reconstructed_alpha_n*np.clip(primal_output1, 0, 1000)).max())
    
#     print(reconstructed_alpha[ALPHA_PC])
    return reconstructed_alpha#, Mat_constr, RHS_, ALPHA_other, ALPHA_0, ALPHA_NC, ALPHA_PC, ALPHA_N0, ALPHA_P0


def primal_to_dual_variables_SVC(X, Y, primal_var, intercept, C, tol = 1.e-4, maxiter = 10000, rtol = 1.e-8):

    primal_output = X@primal_var.T + intercept
#     print(primal_output.shape)
    # constraints where alpha = 0
    ALPHA_0 = np.argwhere(Y*primal_output[:, 0] - 1 > 2*tol)[:, 0]
    
    # constraints where alpha = C
    ALPHA_C = np.argwhere(Y*primal_output[:, 0] - 1 < -2*tol)[:, 0]
    
    # others
    ALPHA_other = np.setdiff1d(np.arange(len(X), dtype = np.int64), 
                                         np.union1d(ALPHA_0, ALPHA_C))
     
    reconstructed_alpha = np.zeros(len(X))
    reconstructed_alpha[ALPHA_0] =  0
    reconstructed_alpha[ALPHA_C] =  C
    
    
    if len(ALPHA_other) > 0:
        Xp1 = np.concatenate((X[ALPHA_other, :], np.ones((len(ALPHA_other), 1))), axis = 1)

        RHS_1 = primal_var - C*np.sum(Y[ALPHA_C]*X[ALPHA_C, :].T, axis = 1) 
        RHS_2 = -C*np.sum(Y[ALPHA_C])
        RHS_ = np.append(RHS_1, RHS_2)
        Mat_1 = Y[ALPHA_other]*X[ALPHA_other, :].T
        Mat_2 = Y[ALPHA_other][np.newaxis, :]
        Mat_ = np.concatenate((Mat_1, Mat_2), axis = 0)


        BOUNDS_L = 0*np.ones_like(ALPHA_other)
        BOUNDS_R = C*np.ones_like(ALPHA_other) 

        res = sp.optimize.lsq_linear(Mat_, RHS_, bounds=(BOUNDS_L, BOUNDS_R), lsq_solver = 'exact', tol = 0.01*tol )
        reconstructed_alpha[ALPHA_other] = res.x
    #     print(ALPHA_other.shape, PHI.shape, RHS_.shape)
    #     RHS_ = Y[ALPHA_other]*(Xp1@RHS_)

    #     mv = lambda vec : CG_matvec_product2(vec, Y[ALPHA_other], Xp1)
        # Mat_constr1 = YY[set_alpha_nonboundary]*phi1[set_alpha_nonboundary, :].T
        # Mat_constr2 = YY[set_alpha_nonboundary][np.newaxis, :]
        # Mat_constr = np.concatenate((Mat_constr1, Mat_constr2), axis = 0)
    #     A = LinearOperator((len(ALPHA_other),len(ALPHA_other)), matvec=mv)
    #     sol, info = sp.sparse.linalg.cg(A, RHS_, maxiter=maxiter, rtol =rtol)
    reconstructed_alphan = reconstructed_alpha*Y
    
    warning = 0
    if np.abs(reconstructed_alphan@X - primal_var).max()>tol:
        print('Warning! KKT Conditions |w - sum alpha_i x_i| > tol', 
                 np.abs(reconstructed_alphan@X - primal_var).max())
        warning = 1
        
    if np.abs((reconstructed_alphan).sum())>tol:
        print('Warning! KKT Conditions |  sum alpha_i | > tol', 
              np.abs(reconstructed_alphan.sum()))
        warning = 1
    
    if np.abs(reconstructed_alpha*np.clip(Y*primal_output.T-1, 0, 1000)).max() > tol:
        print('Warning! KKT Conditions |  |alpha_i(...)| | > tol', 
             np.abs(reconstructed_alpha*np.clip(Y*primal_output.T-1, 0, 1000)).max())
        warning = 1
        
#     print(np.clip(Y*primal_output.T-1, 0, 1000).shape, (reconstructed_alpha*np.clip(Y*primal_output.T-1, 0, 1000)).shape)
        
    if warning:
        print('Increase or decrease tolerance')
        
        
        
#     print(sol - res.x)
    return reconstructed_alpha