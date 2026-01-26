import numpy as np
import itertools
import math
from matplotlib import pyplot as plt
from libsvm import svmutil
from PMKL import Transformation

monomials = Transformation.monomials
class Kernel():
    '''
    General Class
    '''
    def __init__(self, x, Lower, Upper, degree):
        self.K = initK(x, Lower, Upper)
        self.Z = monomials(x, degree)
        
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
    subset_Z = Kernel.subset_Z;
    q = len(P)
    K = np.zeros(tempK[1,1].shape) 
    for i in range(1, 1 + subset_Z):
        for j in range(1, 1 + subset_Z):
#             print(i, j, q/2*(i-1), q/2*i, q/2*(j-1), q/2*j)
            K = K+tempK[i,j]*((Kernel.Z)@ P[int(q/2*(i-1)):int(q/2*i),
                                              int(q/2*(j-1)):int(q/2*j)] @Kernel.Z.T); 
    K = 0.5*(K+K.T);        
    return K

def initK(x, a, b, kernel = 'TK'):
    '''
    K = initK(x,y,a,b,num) function takes two matrices of inputs, as well as
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

def TKtest(x,y,Z1,Z2,a,b,P):
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
    q = len(P)
    subset_Z = Kernel.subset_Z;
    for i in range(1, 1 + subset_Z):
        for j in range(1, 1 + subset_Z):
            K = K + Ktemp[i,j]*(Z1@P[int(q/2*(i-1)):int(q/2*i),int(q/2*(j-1)):int(q/2*j)]@Z2.T);
    return K