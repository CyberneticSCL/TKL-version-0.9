
import numpy as np
import numba as nb
from numba import jit, njit, prange

import scipy as sp 
from Fast_TKL import KernelFunctions
from Fast_TKL import Transformation


import itertools
import math
import time  
from tqdm import trange


from scipy.sparse.linalg import eigsh, eigs, LinearOperator

from Fast_TKL import Fast_Ker_Vec

@njit
def compute_submatrix_product_without_r(vector, indeces, inverse_indeces,  chol_diag):
    """ This function computes 1DKernel to Vector product 
    Kernel for sorted data is form K = L chol_diag L^T, L_{ij} = 1 if i < j -- low triangular 
    Inputs
    vector -- vector to multiply                       ndarray of shape (n_samples)
    indeces -- sorted indeces for the data             ndarray of shape (n_samples)
    inverse_indeces -- inverse transformation          ndarray of shape (n_samples) 
    chol_diag  --diagonal of sorted kernel matrix      ndarray of shape (n_samples)
    Outputs
    K@v -- ndarray (n_samples, ) """
    vector2 = vector[indeces]  #O(n)
    vector3 = np.cumsum(vector2[::-1])[::-1] #O(n)
    vector4 = vector3*chol_diag #O(n)
    vector6 = np.cumsum(vector4)  #O(n)
    vector7 = vector6[inverse_indeces] #O(n)
    return vector7 #O(n)


def compute_sub1DKernel_diagonal(data, alpha_l, alpha_r,  b, a):
    """ This function computes Diagonal of Kernel for 1D data
    # k = \int_{z \geq {x, y}, z \leq b} z^{delta_l + delta_r + 1}
    # Inputs:
    # data ndarray of shape (n_samples)
    # delta_l, delta_r -- real integers
    # b > max(data)
    # a < min(data)
    # Outputs
    # Diagonal of Kernel Matrix -- ndarray (n_samples, ) """
    N = len(data)

    monom = alpha_l + alpha_r + 1

    KKTemp  =  data  
    intZZT  = (b**monom - KKTemp**monom)/monom   
#     ZZT = (data[:, np.newaxis]**alpha_l)@(data[np.newaxis, :]**alpha_r)
    Sub_Kernel_Matrix = intZZT
    return Sub_Kernel_Matrix



def eig_of_1d_matrix(data, monomial_i, monomial_j, indeces, rindeces, rank, b, a):
    """ Eigenvalues of 1Dkernel 
    k(x,y) = \int_{z \geq max{x, y}} z^{monomial_i + monomial_j } dz
    Inputs
    data ndarray of shape (n_samples)
    monomial_i, monomial_j -- monomial degree          
    indeces -- sorted indeces for the data in 1dK      ndarray of shape (n_samples)
    inverse_indeces -- inverse transformation in 1dK   ndarray of shape (n_samples) 
    rank       -- number of eigenvectors
    b, a -- Kernel Paramater
    Output
    Eigenvalues, Eigenvectors -- ndarray of shape (n_samples, rank)
    """
#     b = kernel_parameters.b;
#     a = kernel_parameters.a;
    sorted_data = data[indeces]
    diag_of_subK = compute_sub1DKernel_diagonal(sorted_data, monomial_i, monomial_j, b, a ) #O(n)
    chol_factor = diag_of_subK.copy() 
    chol_factor[1:] = chol_factor[1:] - chol_factor[:-1]
    
    matvecproduct = lambda x: compute_submatrix_product_without_r(x, indeces, rindeces,  chol_factor) 
    A = LinearOperator((len(data),len(data)), matvec=matvecproduct )
    eig, vec = eigsh(A, k=rank)

    return eig, vec #O(rn)


@njit((nb.float64[:, :], nb.float64[:, :]),
     fastmath =True,   cache = True)
def numba_cumsum_along_axis_v4(input_array, output_array):
    #     output_array=np.empty_like(input_array) 
#     rest = np.zeros(input_array.shape[1])
    for ind in range(input_array.shape[0]):
        if ind == 0:
            output_array[ind, :] = input_array[ind, :]
        else:
            output_array[ind, :] = output_array[ind-1, :] + input_array[ind, :]
    return 


@njit(nb.float64[:,:](nb.float64[:, :]))
def numba_cumsum_along_axis_v2(input_array):
    output_array=np.empty_like(input_array) 
    rest = np.zeros(input_array.shape[1])
    for ind in range(input_array.shape[0]):
        rest = rest + input_array[ind, :]
        output_array[ind, :] = rest 
    return output_array
 
# @njit(nb.float64[:,:](nb.float64[:, :], nb.float64[:]), nopython = True)
def compute_submatrix_product_matrix_par(vector, chol_diag):    
    """ This function computes 1DKernel to Matrix product  
    Kernel for sorted data is form K = L chol_diag L^T, L_{ij} = 1 if i < j -- low triangular 
    Inputs
    vector -- vector to multiply                       ndarray of shape (n_samples)
    indeces -- sorted indeces for the data             ndarray of shape (n_samples)
    inverse_indeces -- inverse transformation          ndarray of shape (n_samples) 
    chol_diag  --diagonal of sorted kernel matrix      ndarray of shape (n_samples)
    Outputs
    Kernel@vector -- ndarray (n_samples, ) """
    
#     start = time.time()
#     vector2 = vector[indeces, :]  # nxr O(rn)
    vector2 = vector  # nxr O(rn)
#     vector3 = np.cumsum(vector2[::-1, :], axis = 0)[::-1] #nxr O(rn)
    vector25 = vector2[::-1, :]
    
#     end   = time.time()
#     forward_indeces_time = end-start
#     
#     start = time.time()
    vector3  = numba_cumsum_along_axis_v2(vector25) #nxr O(rn)
    vector35 = vector3[::-1]
#     end   = time.time()
#     cumsum1_time = end-start
    
#     start = time.time()
    vector36 = vector35.T
    vector37 = vector36*chol_diag
    vector4  = vector37.T#nxr O(rn)
#     end   = time.time()
#     product_time = end-start
#     vector6 = np.cumsum(vector4, axis = 0)  #nxr O(rn)
#     start = time.time()
    vector6 = numba_cumsum_along_axis_v2(vector4)  #nxr O(rn)
#     end   = time.time()
#     cumsum2_time = end-start
#     start = time.time()
    vector7 = vector6#nxr O(rn)
#     end   = time.time()
#     rindeces_time = end-start
#     print(' CUMSUM1 ', cumsum1_time, 
#           ' PD ', product_time, ' CUMSUM2 ', cumsum2_time)
    return vector7 #O(rn)
 
    
@njit(nb.float64[:,:](nb.float64[:, :], nb.float64[:], nb.float64[:, :], nb.float64[:, :]),
     fastmath =True,   cache = True)
def compute_submatrix_product_matrix_v5(vector, chol_diag, output, temp):    
    """ This function computes 1DKernel to Matrix product  
    Kernel for sorted data is form K = L chol_diag L^T, L_{ij} = 1 if i < j -- low triangular 
    Inputs
    vector -- vector to multiply                       ndarray of shape (n_samples)
    indeces -- sorted indeces for the data             ndarray of shape (n_samples)
    inverse_indeces -- inverse transformation          ndarray of shape (n_samples) 
    chol_diag  --diagonal of sorted kernel matrix      ndarray of shape (n_samples)
    Outputs
    Kernel@vector -- ndarray (n_samples, ) """
    
    
    output = vector[::-1, :] #nxr
    numba_cumsum_along_axis_v4(output, temp) #nxr O(rn)
    output = temp[::-1, :] #nxr
    temp = output*chol_diag[:, np.newaxis] #nxr
    numba_cumsum_along_axis_v4(temp, output)  #nxr O(rn) 
    return  output #O(rn)

# @njit(nb.float64[:,:](nb.float64[:, :], nb.float64[:], nb.float64[:, :]), nopython = True)
def compute_submatrix_product_matrix_v3(vector, chol_diag):    
    """ This function computes 1DKernel to Matrix product  
    Kernel for sorted data is form K = L chol_diag L^T, L_{ij} = 1 if i < j -- low triangular 
    Inputs
    vector -- vector to multiply                       ndarray of shape (n_samples)
    indeces -- sorted indeces for the data             ndarray of shape (n_samples)
    inverse_indeces -- inverse transformation          ndarray of shape (n_samples) 
    chol_diag  --diagonal of sorted kernel matrix      ndarray of shape (n_samples)
    Outputs
    Kernel@vector -- ndarray (n_samples, ) """
    
#     start = time.time()
#     vector2 = vector[indeces, :]  # nxr O(rn)
#     vector2 = np.zeros_like(vector)
    vector2 = vector  # nxr O(rn)
#     vector3 = np.cumsum(vector2[::-1, :], axis = 0)[::-1] #nxr O(rn)
    vector2 = vector2[::-1, :]
    
#     end   = time.time()
#     forward_indeces_time = end-start
#     
#     start = time.time()
    vector3 = np.zeros_like(vector2)
#     vector25  = numba_cumsum_along_axis_v2(vector2) #nxr O(rn)
    numba_cumsum_along_axis_v4(vector2, vector3) #nxr O(rn)
#     print(np.abs(vector25 - vector3).max())
    vector2 = vector3[::-1]
#     end   = time.time()
#     cumsum1_time = end-start
    
#     start = time.time()
    vector2 = vector2.T
    vector2 = vector2*chol_diag
    vector2  = vector2.T#nxr O(rn)
#     end   = time.time()
#     product_time = end-start
#     vector6 = np.cumsum(vector4, axis = 0)  #nxr O(rn)
#     start = time.time()
    vector3 = np.zeros_like(vector2)
#     vector25  = numba_cumsum_along_axis_v2(vector2) #nxr O(rn)
    numba_cumsum_along_axis_v4(vector2, vector3) #nxr O(rn)
#     print(np.abs(vector25 - vector3).max())
#     end   = time.time()
#     cumsum2_time = end-start
#     start = time.time()
    vector2 = vector3#nxr O(rn)
#     end   = time.time()
#     rindeces_time = end-start
#     print(' CUMSUM1 ', cumsum1_time, 
#           ' PD ', product_time, ' CUMSUM2 ', cumsum2_time)
    return vector2 #O(rn)


def product_subkernel_otimes_lowdim_par_v2(vector, SlUl, Ul,  chol_diag, vector_out, output, temp, temp2):  
    """ This function computes Matrix to vector produce, where
    Matrix = 1DKernel \otimes  Matrix2 and Matrix2 is low dimensional 
    Matrix2 =  U@S@U^*
    Inputs
    vector -- vector to multiply                       ndarray of shape (n_samples)
    
    SlUl   -- S@U.T for Matrix 2                       ndarray of shape (n_samples, r)
    Sl     -- r eigenvalues of Matrix2                 ndarray of shape (r)
    Ul     -- r eigenvectors of Matrix2                ndarray of shape (n_samples, r) 
    chol_diag  --diagonal of sorted 1DKernel           ndarray of shape (n_samples)
    Outputs
    Matrix@vector -- ndarray (n_samples, ) """
    
    
#     start = time.time()
    temp = vector[:, np.newaxis]*SlUl 
    output = compute_submatrix_product_matrix_v5(temp, chol_diag, output, temp2) #v5 faster for N=30000t 
    temp = output*Ul 
    vector_out= np.sum(temp, axis= 1)  
    return vector_out #O(nr)

def product_subkernel_otimes_lowdim_par(vector, SlUlT, Ul,  chol_diag):  
    """ This function computes Matrix to vector produce, where
    Matrix = 1DKernel \otimes  Matrix2 and Matrix2 is low dimensional 
    Matrix2 =  U@S@U^*
    Inputs
    vector -- vector to multiply                       ndarray of shape (n_samples)
    Sl     -- r eigenvalues of Matrix2                 ndarray of shape (n_samples, r)
    Ul     -- r eigenvectors of Matrix2                ndarray of shape (n_samples, r)
    indeces -- sorted indeces for the data in 1dK      ndarray of shape (n_samples)
    inverse_indeces -- inverse transformation in 1dK   ndarray of shape (n_samples) 
    chol_diag  --diagonal of sorted 1DKernel           ndarray of shape (n_samples)
    Outputs
    Matrix@vector -- ndarray (n_samples, ) """
    
#     start = time.time()
    vector_r = vector[::-1]
    step05= vector*SlUlT 
    step1 = step05.T #nxr O(nr)
#     end   = time.time()
#     time_MATMAT1 =  end-start
#     print(step1.shape)compute_submatrix_product_matrix_v3
#     start = time.time()
#     step2 = np.empty_like(step1)
    step2 =compute_submatrix_product_matrix_par(step1,   chol_diag) #O(nr)
#     step2 = compute_submatrix_product_matrix_v3(step1,   chol_diag)
#     end   = time.time()
#     time_submatrix_product_matrix =  end-start
#     print(step2.shape)
#     start = time.time()
    step3 = step2*Ul #nxr matrix -- O(nr)
#     end   = time.time()
#     time_MATMAT2 =  end-start
# #     print(step3.shape)
#     start = time.time()
    step4 = np.sum(step3, axis= 1) #nx1 matrix -- O(nr)
#     end   = time.time()
#     time_MATSUM =  end-start
#     print('MATMAT1 %.2f' % time_MATMAT1, 'KERMAT %.2f' % time_submatrix_product_matrix, 
#           'MATMAT2 %.2f' % time_MATMAT2, 'MATSUM1 %.2f' % time_MATSUM )
#     print(step4.shape)
    return step4 #O(nr)
                               
       
def eigs_of_Gij_v3(data, D,  monomial_i, monomial_j, indeces, rindeces, rank, b, a):
    """Computes eigenvalues and eigenvectors of G_{ij}
    where
    G_{ij}(x, y) = \prod_{d =1}^D \int_{z \geq {x_d, y_d}, z \leq b} z^{monomial_i[d] + monomial_j[d]} 
    
    k(x,y) = \int_{z \geq max{x, y}} z^{monomial_i + monomial_j } dz
    =============================Inputs=================================
    data ndarray of shape (n_samples, D)
    monomial_i, monomial_j -- monomial degree                                   ndarray of shape (D)      
    indeces         -- sorted indeces for the data along the 1st dimension      ndarray of shape (n_samples, D)
    inverse_indeces -- inverse transformation                                   ndarray of shape (n_samples, D) 
    rank       -- number of eigenvectors to estimate
    b, a       -- Kernel Paramater                                              ndarray of shape (D) 
    b > np.max(data, axis = 0), a < np.min(data, axis = 0)
    
    =============================Output=================================
    Eigenvalues, Eigenvectors -- ndarrays of shape (rank) and (n_samples, rank)
    """
    #     print(D,  monomial_i, monomial_j)
    time_start = time.time()
#     print('Start', time.time())
    for dim_index in (np.arange(D, 0, -1)-1): # loop over d iterations
        d = int(dim_index)
        #         print(d)
        if d == D-1:
#             print(data.shape, indeces.shape, rindeces.shape, rank, b ,a)
            S, U = eig_of_1d_matrix(data[:, d], monomial_i[d], monomial_j[d], 
                                               indeces[:, d], rindeces[:, d], rank, b[d], a[d])  #O(r^2 n )
        else:
            print('Elapsed Time= %.2e' % (time.time() - time_start), 'd=', dim_index)
            # compute fast chol product for d matrix and previous
            subindeces  = indeces[:, d]
            subrindeces = rindeces[:, d]
            sorted_data = data[subindeces, d]

            #             print(data.shape, sorted_data.shape, len(data)  )
            diag_of_subK = compute_sub1DKernel_diagonal(sorted_data, monomial_i[d], monomial_j[d], b[d], a[d] ) #O(n)
            chol_factor = diag_of_subK.copy() 
            chol_factor[1:] = chol_factor[1:] - chol_factor[:-1]
            # v2 faster for large data sets due to parallelization
            # v1 faster for small data sets due to vectorization
            U_permut = U[subindeces, :]
            SUT = (S*U_permut).T
            matvecproduct = lambda x: product_subkernel_otimes_lowdim_par(x, SUT, U_permut, chol_factor) #O(rn)

            #             print((len(sorted_data),len(sorted_data)))
            A = LinearOperator((len(sorted_data),len(sorted_data)), matvec=matvecproduct ) #O(rn)
            Sl, Ul = eigsh(A, k=rank) #O(r O(matvec) + r^2 n)
#             Ul = Ul[subrindeces, :]
            S = Sl.copy()
            U = Ul[subrindeces, :].copy()

    return S, U #O(D r^2 n)


def eigs_of_Gij_v4(data, D,  monomial_i, monomial_j, indeces, rindeces, rank, b, a):
    """Computes eigenvalues and eigenvectors of G_{ij}
    where
    G_{ij}(x, y) = \prod_{d =1}^D \int_{z \geq {x_d, y_d}, z \leq b} z^{monomial_i[d] + monomial_j[d]} 
    
    k(x,y) = \int_{z \geq max{x, y}} z^{monomial_i + monomial_j } dz
    =============================Inputs=================================
    data ndarray of shape (n_samples, D)
    monomial_i, monomial_j -- monomial degree                                   ndarray of shape (D)      
    indeces         -- sorted indeces for the data along the 1st dimension      ndarray of shape (n_samples, D)
    inverse_indeces -- inverse transformation                                   ndarray of shape (n_samples, D) 
    rank       -- number of eigenvectors to estimate
    b, a       -- Kernel Paramater                                              ndarray of shape (D) 
    b > np.max(data, axis = 0), a < np.min(data, axis = 0)
    
    =============================Output=================================
    Eigenvalues, Eigenvectors -- ndarrays of shape (rank) and (n_samples, rank)
    """
    #     print(D,  monomial_i, monomial_j)
    time_start = time.time()
    

    N = data.shape[0]
    temp_arr0 = np.empty(N)
    temp_arr1 = np.empty((N, rank))
    temp_arr2 = np.empty((N, rank))
    temp_arr3 = np.empty((N, rank))

    print('Start', time.time())
    for dim_index in (np.arange(D, 0, -1)-1): # loop over d iterations
        d = int(dim_index)
        #         print(d)
        if d == D-1:
#             print(data.shape, indeces.shape, rindeces.shape, rank, b ,a)
            S, U = eig_of_1d_matrix(data[:, d], monomial_i[d], monomial_j[d], 
                                               indeces[:, d], rindeces[:, d], rank, b[d], a[d])  #O(r^2 n )
        else:
            print('Elapsed Time= %.3f' % (time.time() - time_start), 'd=', dim_index)
            # compute fast chol product for d matrix and previous
            subindeces  = indeces[:, d]
            subrindeces = rindeces[:, d]
            sorted_data = data[subindeces, d]

            #             print(data.shape, sorted_data.shape, len(data)  )
            diag_of_subK = compute_sub1DKernel_diagonal(sorted_data, monomial_i[d], monomial_j[d], b[d], a[d] ) #O(n)
            chol_factor = diag_of_subK.copy() 
            chol_factor[1:] = chol_factor[1:] - chol_factor[:-1]
            # v2 faster for large data sets due to parallelization
            # v1 faster for small data sets due to vectorization
            U_permut = U[subindeces, :]
            SU = (S*U_permut)
            time_start2 = time.time()
#             print('Start EigenDecomp')
            matvecproduct = lambda x: product_subkernel_otimes_lowdim_par_v2(x, SU, U_permut, chol_factor, 
                                                                temp_arr0, temp_arr1, temp_arr2, temp_arr3) #O(rn)

            #             print((len(sorted_data),len(sorted_data)))
            A = LinearOperator((len(sorted_data),len(sorted_data)), matvec=matvecproduct ) #O(rn)
            Sl, Ul = eigsh(A, k=rank) #O(r O(matvec) + r^2 n)
#             Ul = Ul[subrindeces, :]
            S = Sl.copy()
            U = Ul[subrindeces, :].copy()
            
#             print('End EigenDecomp Time=%.3f' % (time.time() - time_start2))
    return S, U #O(D r^2 n)
