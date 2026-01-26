import numpy as np
import itertools
import math
from matplotlib import pyplot as plt
from libsvm import svmutil
import scipy.io
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR
from Fast_TKL import KernelFunctions
from Fast_TKL import Optimization
from Fast_TKL import Transformation
import time
from tqdm import trange

Kernel = KernelFunctions.Kernel
makeK  = KernelFunctions.makeK
TKtest = KernelFunctions.TKtest

findP     = Optimization.findP
findAlpha = Optimization.findAlpha
monomials = Transformation.monomials

class PMKL_v2():
    def __init__(self, C = 10,  degree = 1, bound = 0.1, epsilon = 0.1, maxit = 100, tol = 1.e-6, probability = False, to_print= True):
        '''
        C:      Regularization parameter, smaller values lead to mappings that are more general.
        
        degree:  Degree of the TK Kernel function (delta parameter, gamma = 0)
        
        bound:   Area of integration is [-bound,1+bound]
        
        epsilon: Epsilon parameter for regression problems
        
        maxit:   Maximum number of iterations.
        
        tol:     Tolerance of the TKL optimization algorithm
        
        probability: This must be enabled prior to calling fit, will slow down that method as it internally uses 5-fold cross-validation
        
        to_print:  
        '''
#         if params   == None:
        params = ParamsTK(degree, bound, epsilon, maxit, tol)
            
        self.Params = params
        self.to_print = to_print
        self.prob = probability
        self.Params.C = C

        #### PARAMETERS #####
        #### OPT ####
        
        self.Opt = Opt()


        
    def fit(self, x, y, rank = 500):
        '''
        x:      Inputs to be mapped to outputs y. It should be numpy array (n_samples, n_features)
        y:      Outputs. (n_samples)
        '''
        start = time.time()
        Eps = self.Params.epsilon
        tol = self.Params.tol
        maxit = self.Params.maxit
        if len(np.unique(y)) == 2:
            TKL_type = 'Classification';
        else:
            TKL_type = 'Regression';
        self.Type = TKL_type        
        if len(y.shape) < 2:
            y = y[:, np.newaxis]
            
        if y.shape[0]>1 & y.shape[1] == 1 : 
            self.y = y 
        elif y.shape[1]>1 & y.shape[0] == 1  :
            self.y = y.T
        else:
            error('y must be a vector')
            
        if len(self.y) == x.shape[1] : 
            self.xOld = x.T
        elif len(self.y) == x.shape[0]:
            self.xOld = x
        else:
            error('x and y must contain the same number of points')
        
        
        scaleFactor = MinMaxScaler()
        self.x  = scaleFactor.fit_transform(self.xOld)
        self.scaleFactor = scaleFactor
        self.rank = rank
        num, dim = self.x.shape # Dimension and number of inputs
        
        self.Params.Lower =  self.x.min(axis = 0) - self.Params.bound # Lower bounds of integration
        self.Params.Upper =  self.x.max(axis = 0) + self.Params.bound # Upper bounds of integration
        
        self.Kernel = Kernel(self.x, self.Params.Lower, self.Params.Upper, self.Params.degree)
        self.Kernel.low_rank_kernel_SVD(self.rank)
        self.Params.q = self.Kernel.Z.shape[1]
#         print(self.Kernel.K[1,1].shape, self.Kernel.Z.shape)
        q = self.Params.q 
        self.Params.P = np.eye(q) # Initialize P matrix

        
        if self.Type == 'Classification':
            self.model = SVC(C = self.Params.C, kernel = 'precomputed', probability = self.prob)
        else:
            self.model = SVR(C = self.Params.C, epsilon = self.Params.epsilon, kernel = 'precomputed')
            
        maxit = self.Params.maxit
        go = True
        iteration = 0
        Obj = Optimization.findAlpha_lowRankQP(self, self.Kernel)
        self.Opt.Obj.append(Obj)
        self.Opt.diff = np.abs(self.Opt.Obj[-1]-self.Opt.Obj[-2])/np.abs(self.Opt.Obj[-1] + self.Opt.Obj[-2])*200 
        end = time.time()
        self.parsing_time = end - start
        # Calculates the percent difference between objective values
        if self.to_print:
            print('Iteration   |  Objective   |       Dual Gap      | \n')
            print('------------+--------------+---------------------| \n')

#         return self
        while go:
            iteration = iteration+1;
            Optimization.findP_lowRank(self, self.Kernel) # Updates the P matrix which parameterizes the Positive Matrix Kernel Function.
            if (iteration > maxit) or (np.min([self.Opt.dualGap2[-1],self.Opt.dualGap[-1]]) < self.Params.tol) or (self.Opt.StepLength[-1] < 1/100*self.Params.tol) :
            ### Quits if maxit > current iteration or if the objective function change was too small.
                go = False;
            if self.to_print:
                print('%10d  |  %1.4e  |  %1.4e \n' % (iteration, self.Opt.Obj[-1], self.Opt.dualGap[-1]))
#         Ktrain = makeK(self.Kernel, self.Params.P)
#         self.model.fit(Ktrain, self.y)
        return self
        
    def predict(self, Xtest_old, BATCH_SIZE = 10000):
        '''
        
        '''

        if self.Type == 'Classification':
            alpha = self.Params.alpha*self.y
            alpha_hat = self.Params.eigvec.T@alpha
            alpha_hat = self.Params.eigvec@alpha_hat
        else:
            alpha = self.Params.alpha
            alpha_hat = self.Params.eigvec.T@alpha
            alpha_hat = self.Params.eigvec@alpha_hat
            
        print(alpha_hat.shape)
        xTest = self.scaleFactor.transform(Xtest_old)
        xtrain = self.x
        
        dimx = xtrain.shape[1]
        numx = xtrain.shape[0]
        numtest = xTest.shape[0]
        
        yPred =  np.zeros(len(xTest))
        
        for idx in trange(0, len(xtrain), BATCH_SIZE):
            left = idx
            right = min(idx + BATCH_SIZE, len(xtrain))
            Kt = TKtest(xtrain[left:right, :], xTest, 
                        self.Kernel.Z[left:right, :], monomials(xTest,self.Params.degree),
                        self.Params.Lower, self.Params.Upper, self.Params.P, self.Params.add_poly)
            
#             yPred = yPred + self.Params.alpha[left:right, :].T@Kt
        
#             if self.Type == 'Classification':
#                 yPred = np.sign(yPred)
                
#             if self.Type == 'Classification':
#                 vec = self.Params.alpha[left:right, :]*self.y[left:right]
#                 yPred = yPred + vec.T@Kt - self.Params.rho# + self.model.intercept_
#     #             yPred = np.sign(yPred)

#             else:
            yPred = yPred + alpha_hat[left:right, :].T@Kt - self.Params.rho# + self.model.intercept_

            
        if yPred.shape[0] == 1:
            yPred = yPred[0, :]
        return yPred
    
    def predict_proba(self, Xtest_old):
        '''
        Compute probabilities of possible outcomes for samples in X.
        The model need to have probability information computed at training time: fit with attribute probability set to True.
        
        input:
        X array-like of shape (n_samples, n_features)
        
        output:
        
        '''
        xTest = self.scaleFactor.transform(Xtest_old)
        xtrain = self.x[self.model.support_, :]
        
        dimx = xtrain.shape[1]
        numx = xtrain.shape[0]
        numtest = xTest.shape[0]
        
        Kt = TKtest(xtrain, xTest, 
                    self.Kernel.Z[self.model.support_, :], monomials(xTest,self.Params.degree),
                    self.Params.Lower,self.Params.Upper, self.Params.P)
        yPred = self.model.predict_proba(Kt)
        
#         if self.Type == 'Classification':
#             yPred = np.sign(yPred)
            
        if yPred.shape[0] == 1:
            yPred = yPred[0, :]
        return yPred
        
class ParamsTK():
    '''
    [params] = paramsTK(degree,bound,epsilon,maxit,tol) function takes
    degree, bound, epsilon, maxit, and tol parameters as inputs for TK kernel
    functions.  Entering an empty array as an input means the default value
    will be used.
    
    INPUT
    degree:  Degree of the TK Kernel function (delta parameter, gamma = 0)
    bound:   Area of integration is [-bound,1+bound]
    epsilon: Epsilon parameter for regression problems
    maxit:   Maximum number of iterations.
    tol:     Tolerance of the TKL optimization algorithm.
    
    OUTPUT
    params: A variable containing the saved parameter values to be given to TKL.  
    '''
    def __init__(self, degree = 1, bound = 0.1, epsilon = 0.1, maxit = 100, tol = 1.e-6):
        self.degree = degree
        self.bound = bound
        self.epsilon = epsilon
        self.maxit = maxit
        self.tol = tol

        self.kernel = 'TK' 
        self.add_poly = True
        
class Opt():
    def __init__(self):
        self.StepLength = []
        self.dualGap = []
        self.dualGap2 = [np.inf]
        self.l  = [-np.inf  ]
        self.Obj = [-np.inf ]
        self.StepLength.append(1)
        self.TT = [] 
        
        
def loadex1():
    mat = scipy.io.loadmat('PMKLpy/CircleData.mat')['Data']
    X = mat[0,0][0]
    y = mat[0,0][1]
    return X, y


        