import numpy as np
import itertools
import math
from matplotlib import pyplot as plt
from libsvm import svmutil
from PMKL_v2 import Transformation
import Low_Rank_Kernel_Decomp
import time
monomials = Transformation.monomials



class Kernel():
    '''
    General Class
    '''
    def __init__(self, x, Lower, Upper, degree):
#         self.K = initK(x, Lower, Upper)
        self.Z = monomials(x, degree)
        self.b = Upper
        self.a = Lower
        self.x = x
        self.degree = degree
        
        
    def low_rank_kernel_SVD(self, rank):
        deg1 = 0
        deg2 = self.degree
        X = self.x

        N_d = X.shape[-1]
        monomial_index_1 = np.array(list(itertools.product(list(range(deg1+1)), repeat = N_d)))
        monomial_index_1 = monomial_index_1[monomial_index_1.sum(axis = -1) <= deg1][:,::-1]
        # Z1 = monomials(Udata, deg_1)

        monomial_index_2 = np.array(list(itertools.product(list(range(deg2+1)), repeat = N_d)))
        monomial_index_2 = monomial_index_2[monomial_index_2.sum(axis = -1) <= deg2][:,::-1]

        
        n_1 = len(monomial_index_1)
        n_2 = len(monomial_index_2)
#         print(n_1, n_2)
        self.index_G = np.kron(np.ones(n_2), np.arange(n_1)).astype(np.int32)
        self.index_Z = np.kron(np.arange(n_2), np.ones(n_1)).astype(np.int32)
        
#         print(self.index_G, self.index_Z)
        self.Forward_Inds = np.argsort(X, axis=0)[::-1, :]
        self.Reverse_Inds = np.argsort(self.Forward_Inds, axis = 0)
        
        start = time.time()
        # decomposition of low rank G_{ij}
        
        iterations_n1n1 = list(itertools.product(range(n_1), range(n_1))) 

        eigs_G = {} # O(len(Z)^2/2 r^3 n d )
        vecs_G = {} # O(len(Z)^2/2 r^3 n d )
        # parallelization to be implemented
        for item in iterations_n1n1:  
        #     break
        #     print(item)
            i = item[0]
            j = item[1]

            if i > j:
                continue 
            eigs_ij, vecs_ij = Low_Rank_Kernel_Decomp.eigs_of_Gij_v3(X, N_d, 
                               monomial_index_1[0, :], monomial_index_1[0, :],
                                   self.Forward_Inds, self.Reverse_Inds, rank, self.b, self.a) #v4 unstable (memory management error)
            eigs_G[i, j] = eigs_ij
            vecs_G[i, j] = vecs_ij

        end = time.time()
        self.eigs = eigs_G
        self.vecs = vecs_G
#         self.kernel_time = end - start
        print('Low Rank Decomposition has been finished in %.1f sec' % (end - start))
        
def makeK( Kernel, P = None):
    '''
    K = makeK(SVM,Kernel) function takes a support vector machine object, 
    and a Kernel object to generate the kernel matrix.

    INPUT
    Kernel: An internal kernel object used for quickly calculating the kernel matrix.
    P:      An optional P argument can be given to be used instead of the P matrix saved in SVM.

    OUTPUT
    K: The kernel matrix. 
    '''
    
    if P is None:
        P = SVM.Params.P
    tempK = Kernel.K 
    q = len(P)
    K = np.zeros(tempK[1,1].shape) 
    for i in range(1, 3):
        for j in range(1, 3):
#             print(i, j, q/2*(i-1), q/2*i, q/2*(j-1), q/2*j)
            K = K+tempK[i,j]*((Kernel.Z)@ P[int(q/2*(i-1)):int(q/2*i),
                                              int(q/2*(j-1)):int(q/2*j)] @Kernel.Z.T); 
    K = 0.5*(K+K.T);        
    return K

def initK(x, a, b, kernel = 'TK'):
    '''
    K = initK(x,a,b,num) function takes a matrix of inputs, as well as
    a lower (a) and upper (b) bound over which we integrate and the number of
    inputs.
  
    INPUT
    x:   Matrix of inputs to precompute portions of the kernel matrix. (n_samples, n_features)
    a:   Lower bound of integration for the kernel.
    b:   Upper bound of integration for the kernel.
    num: The number of inputs (equivalent to the size of the Kernel matrix).
 
    OUTPUT
    K: Precomputation of parts of the kernel matrix.
    '''
    if kernel != 'TK':
        print("ERROR:That kernel type has not been included.")
        return -1
    xx = x.T
#     print(a, b)
#     print(x.shape, xx.shape)
    n_features = x.shape[1]
    n_samples  = x.shape[0]
    K = {}
    for i in range(n_features):
        kTemp = np.kron(xx[i,:], np.ones((n_samples, 1)))
        
        if i == 0:
            K[1,1] = b[i] - np.maximum(kTemp, kTemp.T)

            K[1,2] = b[i] - kTemp
            K[2,1] = b[i] - kTemp.T
        else:
            K[1,1] = K[1,1]*(b[i] - np.maximum(kTemp, kTemp.T))
            K[1,2] = K[1,2]*(b[i] - kTemp   )
            K[2,1] = K[2,1]*(b[i] - kTemp.T )

#         print(K[1,1][:3, :3])
    K[1,2] = K[1,2] - K[1,1]
    K[2,1] = K[2,1] - K[1,1]
#     print(K[1,1].shape, K[1,2].shape, K[2,1].shape, a.shape, b.shape)
    K[2,2] = np.prod( b-a) - K[1,1] - K[1,2] - K[2,1]
    
    return K    

def compute_D_matrix(alpha_variables, eigvals_of_Gij, vectors_of_Gij, monomials_V, index_Z, index_G):
    
    n1 = np.max(index_G)
    n2 = np.max(index_Z)
    n1p = n1+1
    n2p = n2+1
    output = np.zeros((n1p*n2p, n1p*n2p)) 
    iterations_n1n1 = list(itertools.product(range(n1p), range(n1p))) 
    iterations_n2n2 = list(itertools.product(range(n2p), range(n2p))) 

    monomials_times_vector = (alpha_variables*monomials_V.T).T #n times n2
#     print(monomials_times_vector.shape)
    for item in  iterations_n1n1:  

        i = item[0]
        j = item[1]
        
        ind_min = min(i, j)
        ind_max = max(i, j) 
  
        subI1 = np.argwhere(index_G == i)[:, 0]
        subI2 = np.argwhere(index_G == j)[:, 0]
#         print(subI1, subI2)
        sub_inds = np.ix_(subI1, subI2)
        Gij_vec = vectors_of_Gij[ind_min, ind_max]# matrix nxr
        Gij_val = eigvals_of_Gij[ind_min, ind_max]# matrix n
        
        step1 = Gij_vec.T@monomials_times_vector #r times n2
#         print(step1.shape)
        step2 = (step1.T*Gij_val).T #r times n2
#         print(step2.shape)
        step3 = Gij_vec@step2 #  n times n2   [:, l] is a product Gij@Diag(monomial_l)@vector
         
        step4 = monomials_times_vector.T@step3  
        output[sub_inds] = step4 
    return output



def TKtest(x,y,Z1,Z2,a,b,P, add_poly = False):
    '''
    [K] = TKtest(x,y,Z1,Z2,a,b,num) function takes two matrices of inputs,
    as well as monomial basis of the inputs (Z1,Z2) and a lower (a) and upper
    (b) bound over which we integrate, the number of training inputs (numx), 
    the number of test inputs (numtest) and a matrix P that parameterizes the
    TK kernel function.  Computes the test kernel matrix for a TK kernel.

    INPUT
    x:       Matrix of inputs to precompute portions of the kernel matrix.
    y:       Matrix of inputs to precompute portions of the kernel matrix.
    Z1:      Monomial basis of the training inputs.
    Z2:      Monomial basis of the test inputs.
    a:       Lower bound of integration for the TK kernel.
    b:       Upper bound of integration for the TK kernel.
    P:       The P matrix which parameterizes the TK kernel function.

    OUTPUT
    K: The test kernel matrix.  
    '''

    numx    = x.shape[0]
    numtest = y.shape[0]
    Ktemp = {}
    dim = x.shape[1];
    for n in range(dim):
        kTemp1 = np.kron(x[:,n][:, np.newaxis],np.ones((1,numtest)))
        kTemp2 = np.kron(y[:,n][:, np.newaxis],np.ones((1,numx))).T
        if n == 0:
            Ktemp[1,1] = b[n] - np.maximum(kTemp1,kTemp2)
            Ktemp[1,2] = b[n] - kTemp1
            Ktemp[2,1] = b[n] - kTemp2
        else:
            Ktemp[1,1] = Ktemp[1,1]*(b[n] - np.maximum(kTemp1,kTemp2))
            Ktemp[1,2] = Ktemp[1,2]*(b[n] - kTemp1)
            Ktemp[2,1] = Ktemp[2,1]*(b[n] - kTemp2)
        
#         print(kTemp2.shape, kTemp1.shape, np.maximum(kTemp1,kTemp2).shape, Ktemp[1,1].shape)
    Ktemp[1,2] = Ktemp[1,2] - Ktemp[1,1];
    Ktemp[2,1] = Ktemp[2,1] - Ktemp[1,1];
    Ktemp[2,2] = np.prod(b-a) - Ktemp[1,1] - Ktemp[1,2] - Ktemp[2,1];

    K = np.zeros((numx,numtest))
    q = 2*len(P)
    for i in range(1, 2):
        for j in range(1, 2):
            K = K + Ktemp[i,j]*(Z1@P[int(q/2*(i-1)):int(q/2*i),int(q/2*(j-1)):int(q/2*j)]@Z2.T);
            
    if add_poly:
        K = Z1@Z2.T
    return K


def fast_full_kernel_vector_v2(vector, eigvals_of_Gij, vectors_of_Gij, monomials_V, index_Z, index_G, P, add_poly = False):
    output = 0
    
    n1 = np.max(index_G)
    n2 = np.max(index_Z)
    n1p = n1+1
    n2p = n2+1
#     iterations_P = list(itertools.product(range(len(P)), range(len(P))))
#     error_kernel_approximation = []
#     approximated_kernels = [] 
    output = 0
    
    iterations_n1n1 = list(itertools.product(range(n1p), range(n1p))) 
    iterations_n2n2 = list(itertools.product(range(n2p), range(n2p))) 

    monomials_times_vector = (vector*monomials_V.T).T #n times n2

    if add_poly:
        output = monomials_V@monomials_times_vector.sum(axis = 0)
#     print(monomials_times_vector.shape)
    for item in  iterations_n1n1:  

        i = item[0]
        j = item[1]
        
        ind_min = min(i, j)
        ind_max = max(i, j) 
  
        subI1 = np.argwhere(index_G == i)[:, 0]
        subI2 = np.argwhere(index_G == j)[:, 0]
#         print(subI1, subI2)
        Gij_vec = vectors_of_Gij[ind_min, ind_max]# matrix nxr
        Gij_val = eigvals_of_Gij[ind_min, ind_max]# matrix n
        
        step1 = Gij_vec.T@monomials_times_vector #r times n2
#         print(step1.shape)
        step2 = (step1.T*Gij_val).T #r times n2
#         print(step2.shape)
        step3 = Gij_vec@step2 #  n times n2   [:, l] is a product Gij@Diag(monomial_l)@vector
        
        subI1 = np.argwhere((index_G == i) )[:, 0]
        subI2 = np.argwhere((index_G == j))[:, 0]
        sub_P = P[subI1, :][:,  subI2] #n2 times n2 submatrix
        step45 = step3@sub_P.T
        step55 = step45*monomials_V
        step65 = np.sum(step55, axis = 1)
#         print(step3.shape)
#         for item2 in iterations_n2n2:
            
#             i2 = item2[0]
#             j2 = item2[1]

#             subI1_v = np.argwhere((index_G == i) & (index_Z == i2) )[:, 0]
#             subI2_v = np.argwhere((index_G == j) & (index_Z == j2) )[:, 0]
#             sub_value_P = P[subI1_v,  subI2_v] #n2 times n2 submatrix
        
#             print(subI1_v, subI2_v, sub_value_P)
#             step4 = step3[:, j2]*sub_value_P # n times n2
#     #         print(monomials_V.shape, step4.shape, sub_P.shape)
#             step5 = monomials_V[:, i2]*step4
# #         step6 = np.sum(step5, axis = 1)
# #         print(step5.shape, step4.shape, step3.shape)
#             output = output + step5
    
#         print(np.abs(step65 - output).max())
#         break
        output = output + step65
    return output