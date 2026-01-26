from PMKL import Transformation
import math
import itertools 
import time  
from scipy.special import factorial
from scipy.sparse.linalg import eigsh
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.stats import multivariate_normal
import numpy as np 
import scipy as sp
import numba as nb
from numba import jit
from tqdm import trange, tqdm


monomials = Transformation.monomials
def compute_block_diagonal(structured_data, P, deg1, deg2, b, a):
    ## Compute monomial degrees N(x, z) = Z_{deg = deg2}(x) otimes Z_{deg = deg1}(z)
    monomial_index_Z1 = np.array(list(itertools.product(list(range(deg1+1)), repeat = structured_data.shape[-1])))
    monomial_index_Z1 = monomial_index_Z1[monomial_index_Z1.sum(axis = -1) <= deg1][:,::-1]
#     Z1 = monomials(structured_data, deg_1)

    monomial_index_Z2 = np.array(list(itertools.product(list(range(deg2+1)), repeat = structured_data.shape[-1])))
    monomial_index_Z2 = monomial_index_Z2[monomial_index_Z2.sum(axis = -1) <= deg2][:,::-1]
    Z2_ = monomials(structured_data, deg2)

    size_of_Z1 = len(monomial_index_Z1)
    size_of_Z2 = len(monomial_index_Z2)
    
    Pindeces_z1 = np.kron(np.ones(size_of_Z2), np.arange(size_of_Z1))
    Pindeces_z2 = np.kron(np.arange(size_of_Z2), np.ones(size_of_Z1))
    block_diag_elements = np.zeros((len(structured_data), size_of_Z2, size_of_Z2))
    for i in range(size_of_Z1):
        for j in range(size_of_Z1):
            ind_Pi = np.where(np.abs(Pindeces_z1 - i) < 1.e-5)[0]
            ind_Pj = np.where(np.abs(Pindeces_z1 - j) < 1.e-5)[0]
#             print(ind_Pi, ind_Pj)
            submatrix_P = P[ind_Pi, :][:, ind_Pj]
#             print(submatrix_P.shape)
            monom = monomial_index_Z1[i] + monomial_index_Z1[j] + 1 

            XXX = b**monom - structured_data**monom
            upd = (np.ones((len(structured_data), size_of_Z2, size_of_Z2 ))*submatrix_P)*np.prod(XXX, axis = 1)[:, np.newaxis, np.newaxis]/np.prod(monom)

            block_diag_elements += upd
#         break
#     break
    return block_diag_elements, Z2_, monomial_index_Z1, monomial_index_Z2



def compute_Kernel_Matrix(data, P, deg1, deg2, b, a):
    dim = data.shape[-1]
    N = data.shape[0]

    monomial_Z1 = np.array(list(itertools.product(list(range(deg1+1)), repeat = dim)))
    monomial_Z1 = monomial_Z1[monomial_Z1.sum(axis = -1) <= deg1][:,::-1]
 
    monomial_Z2 = np.array(list(itertools.product(list(range(deg2+1)), repeat = dim)))
    monomial_Z2 = monomial_Z2[monomial_Z2.sum(axis = -1) <= deg2][:,::-1]
    Z2 = monomials(data, deg2)
    n1 = len(monomial_Z1)
    n2 = Z2.shape[-1]
    
#     print(n1, Z2.shape)
    indeces_z1 = np.kron(np.ones(n2), np.arange(n1))
    indeces_z2 = np.kron(np.arange(n2), np.ones(n1))
    
#     print(indeces_z1)    
#     print(indeces_z2)
    Kernel_Matrix = np.zeros((N,N)) 
    
    iteration_list = list(itertools.product(range(n1*n2), range(n1*n2)))
    
    for item in tqdm(iteration_list):  

        i = item[0]
        j = item[1] 

        ind_z1 = int(indeces_z1[i])
        ind_z2 = int(indeces_z1[j])
        ind_v1 = int(indeces_z2[i])
        ind_v2 = int(indeces_z2[j]) 
        monom_ind1 = monomial_Z1[ind_z1]
        monom_ind2 = monomial_Z1[ind_z2] 
        monom = monom_ind1 + monom_ind2 + 1


        intZZT = 1
        for dim_ind in range(dim):
            KKTemp1 =  data[:,dim_ind][:, np.newaxis]@np.ones((1,N)) 
            KKTemp  = np.maximum(KKTemp1,KKTemp1.T)
            intZZT  = intZZT*((b[dim_ind]**monom[dim_ind] - KKTemp**monom[dim_ind])/monom[dim_ind])

        Z2Z2T = Z2[:, ind_v1, np.newaxis]@Z2[:, ind_v2, np.newaxis].T
        Kernel_Matrix = Kernel_Matrix + P[i, j]*Z2Z2T*intZZT

    return Kernel_Matrix

 
def compute_Kernel_Matrix_v2(data, P, deg1, deg2, b, a):
    dim = data.shape[-1]
    N = data.shape[0]

    monomial_Z1 = np.array(list(itertools.product(list(range(deg1+1)), repeat = dim)))
    monomial_Z1 = monomial_Z1[monomial_Z1.sum(axis = -1) <= deg1][:,::-1]
    # Z1 = monomials(Udata, deg_1)
    monomial_Z2 = np.array(list(itertools.product(list(range(deg2+1)), repeat = dim)))
    monomial_Z2 = monomial_Z2[monomial_Z2.sum(axis = -1) <= deg2][:,::-1]
    Z2 = monomials(data, deg2)
    n1 = len(monomial_Z1)
    n2 = Z2.shape[-1]
    
#     print(n1, Z2.shape)
    indeces_z1 = np.kron(np.ones(n2), np.arange(n1))
    indeces_z2 = np.kron(np.arange(n2), np.ones(n1))
    
#     print(indeces_z1)    
#     print(indeces_z2)
    Kernel_Matrix = np.zeros((N,N)) 
    
    iteration_list1 = list(itertools.product(range(n1), range(n1)))
    
    iteration_list2 = list(itertools.product(range(n2), range(n2)))
    
    for item in tqdm(iteration_list1):  

        ind_z1 = item[0]
        ind_z2 = item[1] 

#         ind_z1 = int(indeces_z1[i])
#         ind_z2 = int(indeces_z1[j])
#         ind_v1 = int(indeces_z2[i])
#         ind_v2 = int(indeces_z2[j]) 
        monom_ind1 = monomial_Z1[ind_z1]
        monom_ind2 = monomial_Z1[ind_z2] 
        monom = monom_ind1 + monom_ind2 + 1


        intZZT = 1
        for dim_ind in range(dim):
            KKTemp1 =  data[:,dim_ind][:, np.newaxis]@np.ones((1,N)) 
            KKTemp  = np.maximum(KKTemp1,KKTemp1.T)
            intZZT  = intZZT*((b[dim_ind]**monom[dim_ind] - KKTemp**monom[dim_ind])/monom[dim_ind])
        
        subI1 = np.argwhere(indeces_z1 == ind_z1)[:, 0]
        subI2 = np.argwhere(indeces_z1 == ind_z2)[:, 0]
#         print(subI1, subI2)
        subP = P[ subI1, :][:, subI2]
#         for item_v in iteration_list2:
#             ind_v1 = item_v[0]
#             ind_v2 = item_v[1]
#             Z2Z2T = Z2[:, ind_v1, np.newaxis]@Z2[:, ind_v2, np.newaxis].T
            
#             i = np.argwhere((indeces_z1 == ind_z1)*(indeces_z2 == ind_v1))[0][0]
#             j = np.argwhere((indeces_z1 == ind_z2)*(indeces_z2 == ind_v2))[0][0]
            
#             subK = subK + P[i, j]*Z2Z2T
        subK = Z2@subP@Z2.T
#         print(subP.shape, Z2.shape)
        Kernel_Matrix = Kernel_Matrix + subK*intZZT

    return Kernel_Matrix



def compute_Kernel_Matrix_v3(data, P, deg1, deg2, b, a):
    dim = data.shape[-1]
    N = data.shape[0]

    monomial_Z1 = np.array(list(itertools.product(list(range(deg1+1)), repeat = dim)))
    monomial_Z1 = monomial_Z1[monomial_Z1.sum(axis = -1) <= deg1][:,::-1]
    # Z1 = monomials(Udata, deg_1)
    monomial_Z2 = np.array(list(itertools.product(list(range(deg2+1)), repeat = dim)))
    monomial_Z2 = monomial_Z2[monomial_Z2.sum(axis = -1) <= deg2][:,::-1]
    Z2 = monomials(data, deg2)
    n1 = len(monomial_Z1)
    n2 = Z2.shape[-1]
    
#     print(n1, Z2.shape)
    indeces_z1 = np.kron(np.ones(n2), np.arange(n1))
    indeces_z2 = np.kron(np.arange(n2), np.ones(n1))
    
#     print(indeces_z1)    
#     print(indeces_z2)
    Kernel_Matrix = np.zeros((N,N)) 
    
    iteration_list1 = list(itertools.product(range(n1), range(n1)))
    
    iteration_list2 = list(itertools.product(range(n2), range(n2)))
    
    for item in tqdm(iteration_list1):  

        ind_z1 = item[0]
        ind_z2 = item[1] 

#         ind_z1 = int(indeces_z1[i])
#         ind_z2 = int(indeces_z1[j])
#         ind_v1 = int(indeces_z2[i])
#         ind_v2 = int(indeces_z2[j]) 
        monom_ind1 = monomial_Z1[ind_z1]
        monom_ind2 = monomial_Z1[ind_z2] 
        monom = monom_ind1 + monom_ind2 + 1


        subdata = np.prod((b**monom - data**monom), axis = 1)/np.prod(monom)
        intZZT = 1
#         for dim_ind in range(dim):
        KKTemp1 = subdata[:, np.newaxis]@np.ones((1,N)) 
        intZZT  = np.minimum(KKTemp1,KKTemp1.T)
#             intZZT  = intZZT*((b[dim_ind]**monom[dim_ind] - KKTemp**monom[dim_ind])/monom[dim_ind])
        
        subI1 = np.argwhere(indeces_z1 == ind_z1)[:, 0]
        subI2 = np.argwhere(indeces_z1 == ind_z2)[:, 0]
#         print(subI1, subI2)
        subP = P[ subI1, :][:, subI2]
#         for item_v in iteration_list2:
#             ind_v1 = item_v[0]
#             ind_v2 = item_v[1]
#             Z2Z2T = Z2[:, ind_v1, np.newaxis]@Z2[:, ind_v2, np.newaxis].T
            
#             i = np.argwhere((indeces_z1 == ind_z1)*(indeces_z2 == ind_v1))[0][0]
#             j = np.argwhere((indeces_z1 == ind_z2)*(indeces_z2 == ind_v2))[0][0]
            
#             subK = subK + P[i, j]*Z2Z2T
        subK = Z2@subP@Z2.T
#         print(subP.shape, Z2.shape)
        Kernel_Matrix = Kernel_Matrix + subK*intZZT

    return Kernel_Matrix



def chol_block_separable(diag, m):
    # This function reconstruct the diagonal of structured matrix Ktemp_UU
    if len(m) == 1: 
        # if the matrix is of form 
        # [A A A]   [1  0 0] [A 0     0] [1 -1 0]
        # [A B B] = [-1 1 0] [0 B-A   0] [0 1 -1]
        # [A B C]   [0 -1 1] [0 0   C-B] [0 0  1]
        chol = diag.copy()
        chol[1:, :] = diag[1:, :] - diag[:-1, :] #compute A, B-A, C-B ...
    else:  
        # if it has higher structure
        current_m = m[0] # choose first grid structure for blocks
        size_of_block = len(diag) // current_m # define size of block
        lst = np.array([(i // size_of_block) for i in range(len(diag))]) # define location of the blocks
#         print(len(diag), size_of_block, lst )
        lp = 0
        for i in range(current_m): 
            l = diag[np.where(lst == i)[0], :] - lp # block - previous block
            LL = chol_block_separable(l, m[1:])  # decrease the grid structure
            if i == 0 :
                chol = LL
            else:
                chol = np.concatenate( (chol, LL ) ) # update
            lp = lp + l # since l = Block - lp => lp := Block
    return chol



# def construct_Linv_matrix(d):
#     L = np.eye(d)
#     L[np.arange(1, d), np.arange(d-1)] = -1
#     return L

# def construct_L_matrix(d):
#     L = np.ones((d, d))
#     L = np.tril(L) 
#     return L

# def chol_separable(diag, m):
#     # This function reconstruct the diagonal of structured matrix Ktemp_UU
#     if len(m) == 1: 
#         chol = diag.copy()
#         chol[1:] = diag[1:] - diag[:-1]
#     else:  
#         current_m = m[0]
#         size_of_block = len(diag) // current_m
#         lst = np.array([(i // size_of_block) for i in range(len(diag))])
#         lp = 0
#         for i in range(current_m): 
#             l = diag[np.where(lst == i)[0]] - lp
#             LL = chol_separable(l, m[1:]) 
#             if i == 0 :
#                 chol = LL
#             else:
#                 chol = np.concatenate( (chol, LL ) ) 
#             lp = lp + l 
#     return chol



# def kronmult(Q, x):
#     # https://www.mathworks.com/matlabcentral/fileexchange/23606-fast-and-efficient-kronecker-multiplication
#     # KRONMULT Efficient Kronecker Multiplication 
#     #  Copyright, Stanford University, 2009
#     #  Paul G. Constantine, David F. Gleich
#     N = len(Q) # number of matrices
#     n = np.zeros((N,1))
#     X = np.copy(x)
#     nright = 1
#     nleft = 1
#     for i in range(N-1):
#         n[i] = Q[i].shape[0]
#         nleft = int(nleft*n[i])
        
# #     print(nleft, nright)
#     n[-1] = Q[-1].shape[0]    
#     for i in (np.arange(N, 0, -1)-1):
#         base = 0
#         jump = n[i]*nright;
#         for k in range(nleft):
#             for j  in range(nright):
#                 index1 = base+j
#                 index2 = base+j+nright*(n[i]-1)
#                 inds = np.arange(index1, index2+1, nright, dtype = np.int32)
# #                 print(inds)
#                 X[inds] = Q[i]@X[inds]
#             base = base+jump;
#         nleft = int(nleft/n[max(i-1,0)]);
#         nright =int(nright*n[i]);
#     return X

def kronmult_block_step1(grid_structure, n_2, x):
    # This function multiplies by L1^T \otimes ... \otimes Ln^T \otimes I_n2
    local_grid = grid_structure.copy()
    local_grid.append(n_2)
    N = len(local_grid) # number of matrices
    n = np.zeros((N,1))
    X = np.copy(x)
    nright = 1
    nleft = 1
    n = local_grid
    nleft = int(np.prod(n))
    nleft = nleft//local_grid[-1] 
    
    for i in (np.arange(N, 0, -1)-1):
        base = 0
        jump = n[i]*nright;
        for k in range(nleft):
            
            for j  in range(nright):
                index1 = base+j
                index2 = base+j+nright*(n[i]-1)
                inds = np.arange(index1, index2+1, nright, dtype = np.int32)
                # print(inds)
                # multiplication by Lni^T
                chosen_X = X[inds]
                if i != len(grid_structure):
                    X[inds] = np.cumsum(chosen_X[::-1])[::-1] 
                
            base = base+jump;
        nleft = int(nleft/n[max(i-1,0)]);
        nright =int(nright*n[i]);
    return X


def kronmult_block_step1_v2(grid_structure, n_2, x):
    # This function multiplies by L1^T \otimes ... \otimes Ln^T \otimes I_n2
    local_grid = grid_structure.copy()
#     local_grid.append(n_2)
    N = len(local_grid) # number of matrices
    n = np.zeros((N,1))
    X = np.copy(x)
    nright = 1
    nleft = 1
    n = local_grid
    nleft = int(np.prod(n))
    nleft = nleft//local_grid[-1] 
    
    for i in (np.arange(N, 0, -1)-1):
        base = 0
        jump = n[i]*nright;
        for k in range(nleft):
            
            for j  in range(nright):
                index1 = base+j
                index2 = base+j+nright*(n[i]-1)
                inds = np.arange(index1, index2+1, nright, dtype = np.int32)
                # print(inds)
                # multiplication by Lni^T
                chosen_X = X[inds, :]
                if i != len(grid_structure):
                    X[inds] = np.cumsum(chosen_X[::-1, :], axis = 0)[::-1, :] 
                
            base = base+jump;
        nleft = int(nleft/n[max(i-1,0)]);
        nright =int(nright*n[i]);
    return X

def kronmult_block_step2(grid_structure, n_2, x):
    # This function multiplies by L1 \otimes ... \otimes Ln \otimes I_n2
    local_grid = grid_structure.copy()
    local_grid.append(n_2)
    N = len(local_grid) # number of matrices
    n = np.zeros((N,1))
    X = np.copy(x)
    nright = 1
    nleft = 1
    n = local_grid
    nleft = int(np.prod(n))
    nleft = nleft//local_grid[-1] 
    
    for i in (np.arange(N, 0, -1)-1):
        base = 0
        jump = n[i]*nright;
#         iterlist = itertools.product(range(nleft), range(n))
        for k in range(nleft):
            
            for j  in range(nright):
                index1 = base+j
                index2 = base+j+nright*(n[i]-1)
                inds = np.arange(index1, index2+1, nright, dtype = np.int32)
#                 print(inds)
                # multiplication by Lni^T
                chosen_X = X[inds]
        
                if i != len(grid_structure):
                    X[inds] = np.cumsum(chosen_X)
    
            base = base+jump;
        nleft = int(nleft/n[max(i-1,0)]);
        nright= int(nright*n[i]);
    return X

def kronmult_block_step2_v2(grid_structure, n_2, x):
    # This function multiplies by L1 \otimes ... \otimes Ln \otimes I_n2
    local_grid = grid_structure.copy()
#     local_grid.append(n_2)
    N = len(local_grid) # number of matrices
    n = np.zeros((N,1))
    X = np.copy(x)
    nright = 1
    nleft = 1
    n = local_grid
    nleft = int(np.prod(n))
    nleft = nleft//local_grid[-1] 
    
    for i in (np.arange(N, 0, -1)-1):
        base = 0
        jump = n[i]*nright;
#         iterlist = itertools.product(range(nleft), range(n))
        for k in range(nleft):
            
            for j  in range(nright):
                index1 = base+j
                index2 = base+j+nright*(n[i]-1)
                inds = np.arange(index1, index2+1, nright, dtype = np.int32)
#                 print(inds)
                # multiplication by Lni^T
                chosen_X = X[inds, :]
        
                if i != len(grid_structure):
                    X[inds, :] = np.cumsum(chosen_X, axis = 0)
    
            base = base+jump;
        nleft = int(nleft/n[max(i-1,0)]);
        nright= int(nright*n[i]);
    return X

def Block_Kernel_to_Vec_Multiplication_v2(diag, grid_structure, x):
    size_of_D = diag.shape[-1]
    X1 = kronmult_block_step1_v2(grid_structure, size_of_D, x)
    

#     X1_reshaped = X1.reshape(-1, size_of_D)
    X2 = np.einsum('lij,lj->li', diag, X1) 
#     X2 = X2_reshaped.reshape(-1)
    
    X3 = kronmult_block_step2_v2(grid_structure, size_of_D, X2)
    return X3


def Block_Kernel_to_Vec_Multiplication(diag, grid_structure, x):
    size_of_D = diag.shape[-1]
    X1 = kronmult_block_step1(grid_structure, size_of_D, x)
    

    X1_reshaped = X1.reshape(-1, size_of_D)
    X2_reshaped = np.einsum('lij,lj->li', diag, X1_reshaped) 
    X2 = X2_reshaped.reshape(-1)
    
    X3 = kronmult_block_step2(grid_structure, size_of_D, X2)
    return X3



def Fast_Kernel_to_Vec_Multiplication(diag, Monomials, grid_structure, x):
    size_of_D = diag.shape[-1] 
    X1_reshaped = Monomials.T*x  
    X1 = X1_reshaped.T.reshape(-1) 
    X2 = Block_Kernel_to_Vec_Multiplication(diag, grid_structure, X1) 
    # w2_reshaped = w2.reshape(-1, n_2)
    X3_b = (Monomials.reshape(-1)*X2).reshape(-1, size_of_D) 
    X3  = np.sum(X3_b, axis = -1)  
    return X3

def Fast_Kernel_to_Vec_Multiplication_v2(diag, Monomials, grid_structure, x):
    size_of_D = diag.shape[-1] 
    X1 = Monomials.T*x  
#     print(X1.shape)
#     X1 = X1_reshaped.T.reshape(-1) 
    X2 = Block_Kernel_to_Vec_Multiplication_v2(diag, grid_structure, X1.T) 
    # w2_reshaped = w2.reshape(-1, n_2)
#     print(X2.shape)
    X3_b = Monomials*X2
#     X3_b = (Monomials.reshape(-1)*X2.reshape(-1)).reshape(-1, n_2) 
#     print(np.abs(X3_b-X3_bp).max())
    X3  = np.sum(X3_b, axis = -1)  
    return X3

def MV_gaussian_pdf(x, mu, cov):
    
    dim = cov.shape[-1]
    C = np.sqrt(np.linalg.det(cov)*(2*np.pi)**dim)
    f1 =  np.linalg.inv(cov)@(x - mu).T
    f2 = -0.5*((x-mu)*f1.T).sum(axis = -1)
    f3 = np.exp(f2)/C
    
    return f3

def sqrt_of_MV_gaussian_pdf(x, mu, cov):
    
    dim = cov.shape[-1]
    C = np.sqrt(np.sqrt(np.linalg.det(cov)*(2*np.pi)**dim))
    f1 =  np.linalg.inv(cov)@(x - mu).T
    f2 = -0.25*((x-mu)*f1.T).sum(axis = -1)
    f3 = np.exp(f2)/C
    
    return f3


def Block_eigenvalues(structured_data, grid_structure, P, deg1, deg2, k_eigs, mu = 0, cov = 1, b = 1, a = 0):
    
    diag_of_K, monomials, mon_ind1, mon_ind2 = compute_block_diagonal(structured_data, P, deg1, deg2, b, a) # diagonal of structured kernel
#     print(diag_of_K.shape)
    D_of_Kernel = chol_block_separable(diag_of_K, grid_structure)  # cholesky decomposition
#     print(D_of_Kernel.shape)
    prob_of_data = sqrt_of_MV_gaussian_pdf(structured_data, mu, cov)  #probability of the data
    
    Kernel_rho_vec = lambda v: prob_of_data*Fast_Kernel_to_Vec_Multiplication_v2(D_of_Kernel, monomials,
                                                                              grid_structure, prob_of_data*v)
    
#     print(np.prod(grid_structure))
    A = LinearOperator((np.prod(grid_structure),np.prod(grid_structure)), matvec=Kernel_rho_vec) # linear operator for EigDec
   
    lambdas, vectors = eigsh(A, k=k_eigs)
    
    vectors_out = vectors[:,::-1]
    lambdas_out = lambdas[::-1]
    
    return lambdas_out, vectors_out
# D_of_Kernel = chol_separable(diag_of_Ktemp2, grid_struct)
# D_of_Kernel_nonuni = chol_separable(diag_of_Ktemp_nonuni, grid_struct)
# KernelVec = lambda v: Kernel_to_Vec_Multiplication(D_of_Kernel, grid_struct, v)
# KernelVec_nonuni = lambda v: Kernel_to_Vec_Multiplication(D_of_Kernel_nonuni, grid_struct, v)
